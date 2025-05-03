#!/usr/bin/env bash
set -euo pipefail

ROLE=${ROLE:-"worker"}            # "head" or "worker"
RAY_PORT=6379

if [[ "$ROLE" == "head" ]]; then
  ray start --head --port=${RAY_PORT} --dashboard-port=8265 \
            --resources='{"gpu":'"${NUM_GPUS:-1}"'}' \
            --block &          # stays in foreground
  sleep 5                      # give the head some time
  python -m vllm.entrypoints.openai.api_server \
       --model ${MODEL_ID:-"NousResearch/Meta-Llama-3-8B-Instruct"} \
       --tensor-parallel-size ${TP_SIZE:-2} \
       --pipeline-parallel-size ${PP_SIZE:-1} \
       --engine-use-ray true \
       --host 0.0.0.0 --port 8000
else
  ray start --address=${RAY_HEAD_ADDR}:${RAY_PORT} \
            --resources='{"gpu":'"${NUM_GPUS:-1}"'}' --block
fi
