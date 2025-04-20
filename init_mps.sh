export CUDA_VISIBLE_DEVICES=0 # In image building
nvidia-cuda-mps-control -d # start nvidia-cuda-mps
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps" # In image building
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 # Specify when run container 
export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=5GB" # Specify when run container
