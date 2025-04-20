#sudo podman run --rm -it --device nvidia.com/gpu=all --security-opt=label=disable -v /home/student.unimelb.edu.au/bxb1/Megatron-LM/megatron:/workspace/megatron -v /home/student.unimelb.edu.au/bxb1/checkpoints:/workspace/checkpoints -v /home/student.unimelb.edu.au/bxb1/EE-LLM:/workspace/EE-LLM  nvcr.io/nvidia/pytorch:22.12-py3
# sudo podman run -dit \
#   --device nvidia.com/gpu=all \
#   --security-opt=label=disable \
#   -v /home/student.unimelb.edu.au/bxb1/Megatron-LM/megatron:/workspace/megatron \
#   -v /home/student.unimelb.edu.au/bxb1/checkpoints:/workspace/checkpoints \
#   -v /home/student.unimelb.edu.au/bxb1/EE-LLM:/workspace/EE-LLM \
#   --name ee-llm-dev \
#   nvcr.io/nvidia/pytorch:22.12-py3 \
#   bash

# pip install flask flask_restful sentencepiece
# python setup.py install

sudo podman run -dit \
  --ipc=host \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  -v /home/student.unimelb.edu.au/bxb1/assets:/workspace/assets \
  -v /home/student.unimelb.edu.au/bxb1/EE-LLM:/workspace/EE-LLM \
  --name stage1 \
  ee-llm:latest \
  bash
  # --shm-size=96g \

# sudo docker run -dit \
#   --gpus all \
#   --rm \
#   --shm-size=96gb \
#   -v /home/student.unimelb.edu.au/bxb1/Megatron-LM/megatron:/workspace/megatron \
#   -v /home/student.unimelb.edu.au/bxb1/checkpoints:/workspace/checkpoints \
#   -v /home/student.unimelb.edu.au/bxb1/EE-LLM:/workspace/EE-LLM \
#   --name ee-llm-dev \
#   nvcr.io/nvidia/pytorch:22.12-py3 \
#   bash
