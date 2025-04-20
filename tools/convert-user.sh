#!/bin/bash

LOAD_DIR=/workspace/checkpoints/EE-LLM-1B-dj-refine-300B
SAVE_DIR=/workspace/checkpoints/EE-LLM-1B-dj-refine-300B/convert-2

TP=1
PP=2
CUR_DIR=$(cd $(dirname "$0") && pwd)
MEGATRON_ROOT_PATH=$(cd "$CUR_DIR/.." && pwd)
cd $MEGATRON_ROOT_PATH

python $MEGATRON_ROOT_PATH/tools/checkpoint/util.py --model-type EarlyExitGPT --load-dir $LOAD_DIR --save-dir $SAVE_DIR --target-tensor-parallel-size $TP --target-pipeline-parallel-size $PP --megatron-path $MEGATRON_ROOT_PATH

