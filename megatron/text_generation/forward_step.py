# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Forward step utilities."""

from collections.abc import Iterable

import torch

from megatron import get_args
from megatron.core import mpu
from .inference_params import InferenceParams
from .communication import (
    send_to_next_pipeline_rank,
    recv_from_prev_pipeline_rank_,
    send_list_to_next_pipeline_rank,
    recv_list_from_prev_pipeline_rank)


class ForwardStep:
    """
    Encapsulates one “step” of model forward execution, handling both
    pipelined communication and early-exit logic for inference.

    Responsibilities:
      1. Manage model.eval() invocation and inference parameter injection.
      2. Route tensors through pipeline stages:
         - recv from previous rank
         - invoke model(tokens, positions, mask, inference_params)
         - send outputs to next rank
      3. Support non-pipelined vs. pipelined micro-batch execution:
         - `_no_pipelining_forward_step()` for single-pass
         - `_with_pipelining_forward_step()` for batch→micro-batch slicing
      4. Integrate early-exit pipelining:
         - `_with_early_exit_pipelining_forward_step()` to recv/send
           early-exit signals alongside activations
      5. Track and update `InferenceParams` offsets:
         - `sequence_len_offset` and `batch_size_offset` for caching
         - propagate `has_early_exited` flags across pipeline

    Usage:
        forward = ForwardStep(model, inference_params=my_params)
        logits = forward(tokens, position_ids, attention_mask)
        # logits is only non-None on the last pipeline stage
    """

    def __init__(self, model, max_batch_size=0, max_sequence_length=0, early_exit_thres=0, inference_params=None):
        """Set values so we don't need to do it multiple times."""
        # Make sure model is in eval mode.
        assert not isinstance(model, Iterable), \
            'interleaving schedule is not supported for inference'
        model.eval()
        self.model = model
        # Initialize inference parameters.
        if inference_params is None:
            self.inference_params = InferenceParams(max_batch_size,
                                                max_sequence_length, early_exit_thres)
        else:
            self.inference_params = inference_params
        # Pipelining arguments.
        args = get_args()
        self.pipeline_size_larger_than_one = (
            args.pipeline_model_parallel_size > 1)
        # Threshold of pipelining.
        self.pipelining_batch_x_seqlen = \
            args.inference_batch_times_seqlen_threshold


    def __call__(self, tokens, position_ids, attention_mask):
        """Invocation of the forward methods. Note that self.inference_params
        is being modified by the forward step."""
        # Pipelining case.
        if self.pipeline_size_larger_than_one:
            return _with_early_exit_pipelining_forward_step(self.model,
                                                     tokens,
                                                     position_ids,
                                                     attention_mask,
                                                     self.inference_params)

        return _no_pipelining_forward_step(self.model,
                                           tokens,
                                           position_ids,
                                           attention_mask,
                                           self.inference_params)



def _get_recv_buffer_dtype(args):
    """Receive happens between the layers."""
    if args.fp32_residual_connection:
        return torch.float
    return args.params_dtype



def _allocate_recv_buffer(batch_size, sequence_length):
    """Receive happens between the layers with size [s, b, h]."""
    if mpu.is_pipeline_first_stage():
        return None
    args = get_args()
    recv_size = (sequence_length, batch_size, args.hidden_size)
    return torch.empty(recv_size,
                       dtype=_get_recv_buffer_dtype(args),
                       device=torch.cuda.current_device())



def _forward_step_helper(model, tokens, position_ids, attention_mask,
                         inference_params, recv_buffer=None):
    """Single forward step. Update the allocate memory flag so
    only the first time the memory is allocated."""
    batch_size = tokens.size(0)
    sequence_length = tokens.size(1)
    if recv_buffer is None:
        recv_buffer = _allocate_recv_buffer(batch_size, sequence_length)

    # Receive from previous stage.
    recv_from_prev_pipeline_rank_(recv_buffer)

    # Forward pass through the model.
    model.set_input_tensor(recv_buffer)
    output_tensor = model(tokens, position_ids, attention_mask,
                          inference_params=inference_params)

    # Send output to the next stage.
    send_to_next_pipeline_rank(output_tensor)

    return output_tensor



def _no_pipelining_forward_step(model, tokens, position_ids, attention_mask,
                                inference_params, recv_buffer=None):
    """If recv_buffer is none, we will allocate one on the fly."""
    # Run a simple forward pass.
    output_tensor = _forward_step_helper(model, tokens, position_ids,
                                         attention_mask, inference_params,
                                         recv_buffer=recv_buffer)
    # Update the sequence length offset.
    # inference_params.sequence_len_offset += tokens.size(1)

    logits = None
    if mpu.is_pipeline_last_stage():
        logits = output_tensor

    return logits



def _with_pipelining_forward_step(model, tokens, position_ids, attention_mask,
                                  inference_params, micro_batch_size):
    """No interleaving is supported."""
    sequence_length = tokens.size(1)
    batch_size = tokens.size(0)

    # Divide the batch dimension into micro batches.
    num_micro_batches, last_chunk = divmod(batch_size,
                                           micro_batch_size)
    if last_chunk > 0:
        num_micro_batches += 1

    # Preallocate memory for output logits.
    logits = None
    if mpu.is_pipeline_last_stage():
        args = get_args()
        logits = torch.empty(
            (batch_size, sequence_length, args.padded_vocab_size),
            dtype=torch.float32, device=torch.cuda.current_device())

    # Preallocate recv buffer.
    recv_buffer = _allocate_recv_buffer(micro_batch_size, sequence_length)

    for micro_batch_index in range(num_micro_batches):
        # Slice among the batch dimenion.
        start = micro_batch_index * micro_batch_size
        end = min(start + micro_batch_size, batch_size)
        this_micro_batch_size = end - start
        tokens2use = tokens[start:end, ...]
        position_ids2use = position_ids[start:end, ...]

        # Run a simple forward pass.
        if this_micro_batch_size != micro_batch_size:
            recv_buffer = None
        output = _forward_step_helper(model, tokens2use, position_ids2use,
                                      attention_mask, inference_params,
                                      recv_buffer=recv_buffer)

        # Adjust the batch size offset to account for the micro-batch.
        inference_params.batch_size_offset += this_micro_batch_size

        # Copy logits.
        if mpu.is_pipeline_last_stage():
            logits[start:end, ...] = output

    # Once we are done with all the micro-batches, we can
    # adjust the sequence length offset.
    inference_params.sequence_len_offset += sequence_length
    # and reset the batch size offset
    inference_params.batch_size_offset = 0

    return logits


def _allocate_early_exit_recv_buffers(batch_size, sequence_length):
    if mpu.is_pipeline_first_stage():
        return None
    args = get_args()
    recv_size = (sequence_length, batch_size, args.hidden_size)
    return [torch.empty(recv_size,
                       dtype=_get_recv_buffer_dtype(args),
                       device=torch.cuda.current_device()),
            torch.empty(1, dtype=torch.int8, device=torch.cuda.current_device())]


def _with_early_exit_pipelining_forward_step(model, tokens, position_ids, attention_mask,
                                  inference_params):
    """
    Perform a single forward pass in a pipeline-parallel setting with early-exit support.

    This helper function:
      1. Receives activations and an early-exit flag from the previous pipeline stage.
      2. Updates `inference_params.prev_has_early_exited` based on the received flag.
      3. Calls the model’s forward method with `inference_params`, which may set
         `inference_params.has_early_exited` if its own early-exit criteria are met.
      4. Builds a 1-element `signal_tensor` encoding whether any stage (previous or current)
         has triggered early-exit.
      5. Sends `[output_tensor, signal_tensor]` to the next pipeline stage.
      6. Increments `inference_params.sequence_len_offset` by the full sequence length.

    Args:
        model (torch.nn.Module):
            The language model already in eval mode.
        tokens (torch.Tensor):
            Input token IDs, shape [batch_size, sequence_length].
        position_ids (torch.Tensor):
            Positional IDs matching `tokens`, same shape.
        attention_mask (torch.Tensor):
            Causal attention mask for the current sequence slice.
        inference_params (InferenceParams):
            Carries sampling hyperparameters, KV‐cache, and early‐exit state.

    Returns:
        torch.Tensor:
            The output activation tensor from this stage, of shape
            [batch_size, sequence_length, hidden_size].

    Raises:
        AssertionError:
            If `batch_size != 1`, since early-exit is currently only supported
            for single-sample inference.

    Communication:
        - Uses `recv_list_from_prev_pipeline_rank` to get `[activations, exit_flag]`
        - Uses `send_list_to_next_pipeline_rank` to forward `[output_tensor, signal_tensor]`
    """
    sequence_length = tokens.size(1)
    batch_size = tokens.size(0)
    assert batch_size == 1, "early exit not support batch inference yet"
    # Divide the batch dimension into micro batches.
    # Preallocate recv buffer.
    if not mpu.is_pipeline_first_stage():
        recv_buffers = _allocate_early_exit_recv_buffers(batch_size, sequence_length)
        recv_list_from_prev_pipeline_rank(recv_buffers)
        model.set_input_tensor(recv_buffers[0])
        inference_params.prev_has_early_exited = bool(recv_buffers[1])
    output_tensor = model(tokens, position_ids, attention_mask, inference_params=inference_params)
    signal_tensor = torch.tensor([int(inference_params.has_early_exited or inference_params.prev_has_early_exited)],
                                 dtype=torch.int8,
                                 device=torch.cuda.current_device())
    send_list_to_next_pipeline_rank([output_tensor, signal_tensor])
    inference_params.sequence_len_offset += sequence_length
    return output_tensor