# #!/bin/bash

# PROJECT_NAME=EE-LLM

# export OMP_NUM_THREADS=8
# export CUDA_DEVICE_MAX_CONNECTIONS=1

# # Tokenizer
# TOKENIZER_PATH=/workspace/checkpoints/EE-LLM-1B-dj-refine-300B/tokenizer.model
# # Checkpoint
# CHECKPOINT_PATH=/workspace/checkpoints/EE-LLM-1B-dj-refine-300B/convert-1
# # Parallelism
# TP=1
# PP=1
# # Server port
# PORT=5000

# MASTER_ADDR=127.0.0.1
# MASTER_PORT=5950
# NPROC_PER_NODE=$(( $TP * $PP ))
# LOAD_ITERATION=0

# DIST_ARGS="
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT \
#     --nproc_per_node $NPROC_PER_NODE \
#     --nnodes 1 \
#     --node_rank 0 \
#     "

# SERVER_ARGS="
#   --use-checkpoint-args \
#   --tokenizer-type SentencePieceTokenizer \
#   --tokenizer-model $TOKENIZER_PATH \
#   --load $CHECKPOINT_PATH \
#   --load-iteration $LOAD_ITERATION \
#   --port $PORT
# "

# CUR_DIR=$(cd $(dirname "$0") && pwd)
# MEGATRON_ROOT_PATH=$(cd "$CUR_DIR/../.." && pwd)
# cd $MEGATRON_ROOT_PATH

# torchrun $DIST_ARGS \
#     tools/run_early_exit_text_generation_server.py \
#     $SERVER_ARGS


#!/bin/bash
# run_server_distributed.sh

PROJECT_NAME=EE-LLM

export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} 

# —— 用户可调参数 —— 
# Tokenizer & Checkpoint 路径（容器内路径）
TOKENIZER_PATH=${TOKENIZER_PATH:?请传入tokenizer的路径}
# /workspace/assets/checkpoints/EE-LLM-1B-dj-refine-300B/tokenizer.model
CHECKPOINT_PATH=${CHECKPOINT_PATH:?请传入checkpoint文件的路径}
# /workspace/assets/checkpoints/EE-LLM-1B-dj-refine-300B/convert-2
# 并行度设置
TP=1                  # Tensor 并行度
PP=2                  # Pipeline 并行度：2 段
# Server 口
PORT=${PORT:-5000}

# 多节点 TorchRun 参数
MASTER_ADDR=${MASTER_ADDR:?请声明Master Address}   # Stage0 容器名或 IP
MASTER_PORT=${MASTER_PORT:?请声明Master Port}
NNODES=2
NODE_RANK=${NODE_RANK:?请传入Node Rank}
NPROC_PER_NODE=1

DIST_ARGS="\
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  --nnodes $NNODES \
  --node_rank $NODE_RANK \
  --nproc_per_node $NPROC_PER_NODE \
"

# Megatron Server 参数
SERVER_ARGS="\
  --tensor-model-parallel-size $TP \
  --pipeline-model-parallel-size $PP \
  --use-checkpoint-args \
  --tokenizer-type SentencePieceTokenizer \
  --tokenizer-model $TOKENIZER_PATH \
  --load $CHECKPOINT_PATH \
  --port $PORT \
"

# 切到项目根目录
CUR_DIR=$(cd $(dirname "$0") && pwd)
MEGATRON_ROOT_PATH=$(cd "$CUR_DIR/../.." && pwd)
cd $MEGATRON_ROOT_PATH

# 最终启动
torchrun $DIST_ARGS \
  tools/run_early_exit_text_generation_server.py \
  $SERVER_ARGS
