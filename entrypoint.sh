#!/usr/bin/env bash

# 添加调试信息
echo "=== 环境变量检查 ==="
echo "TOKENIZER_PATH: ${TOKENIZER_PATH:-未设置}"
echo "CHECKPOINT_PATH: ${CHECKPOINT_PATH:-未设置}"
echo "MASTER_ADDR: ${MASTER_ADDR:-未设置}"
echo "MASTER_PORT: ${MASTER_PORT:-未设置}"
echo "NODE_RANK: ${NODE_RANK:-未设置}"
echo "TP: ${TP:-未设置}"
echo "PP: ${PP:-未设置}"
echo "PORT: ${PORT:-未设置}"
echo "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: ${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-未设置}" 
echo "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: ${CUDA_MPS_PINNED_DEVICE_MEM_LIMIT:-未设置}" 
echo "=== 环境变量检查结束 ==="

# 1. 启动 MPS 控制守护进程

# 2. 给 MPS 一点时间准备管道
# # 3. 根据 STAGE 环境变量选择执行"阶段一"或"阶段二"
# if [ "$STAGE" == "1" ]; then
#     echo ">>> New Entry Point Launching Megatron pipeline stage 1"
#     export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-25}
#     export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=${CUDA_MPS_PINNED_DEVICE_MEM_LIMIT:-"0=5GB"}
#     # 启动您的阶段一服务，例如：
#     # python3 run_stage1.py
# elif [ "$STAGE" == "2" ]; then
#     echo ">>> New Entry Point Launching Megatron pipeline stage 2"
#     export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-25}
#     export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=${CUDA_MPS_PINNED_DEVICE_MEM_LIMIT:-"0=5GB"}
#     # 启动您的阶段二服务，例如：
#     # python3 run_stage2.py
# fi

# tail -f /dev/null

PROJECT_NAME=EE-LLM
set -e

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} 

nvidia-cuda-mps-control -d

# —— 用户可调参数 —— 
# Tokenizer & Checkpoint 路径（容器内路径）
TOKENIZER_PATH=${TOKENIZER_PATH:?请传入tokenizer的路径}
# /workspace/assets/checkpoints/EE-LLM-1B-dj-refine-300B/tokenizer.model
CHECKPOINT_PATH=${CHECKPOINT_PATH:?请传入checkpoint文件的路径}
# /workspace/assets/checkpoints/EE-LLM-1B-dj-refine-300B/convert-2
# 并行度设置
TP=${TP:-1}                  # Tensor 并行度
PP=${PP:-2}                  # Pipeline 并行度：2 段
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

echo $SERVER_ARGS
echo $DIST_ARGS
# 切到项目根目录
# CUR_DIR=$(cd $(dirname "$0") && pwd)
# MEGATRON_ROOT_PATH=$(cd "$CUR_DIR/../.." && pwd)
# cd $MEGATRON_ROOT_PATH

# 最终启动
torchrun $DIST_ARGS \
  /workspace/EE-LLM/tools/run_early_exit_text_generation_server.py \
  $SERVER_ARGS
