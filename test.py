# #!/usr/bin/env python3

# import os
# import sys
# import time
# import subprocess
# import multiprocessing

# # Optional: If you want to hard-set MPS environment variables here, do so.
# # Typically you would set them in your shell or in a systemd unit before launching this script.
# # os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
# # os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-log"
# # os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "50"

# def gpu_worker(worker_id):
#     """
#     Each worker does large matrix multiplications in a loop to saturate the GPU.
#     """
#     import torch  # Import inside the worker to avoid issues on some systems
#     print(f"[Worker {worker_id}] Starting heavy GPU load...")
#     device = torch.device("cuda")  # Force GPU usage

#     while True:
#         # Create random matrices on GPU
#         a = torch.randn((4096, 4096), device=device)
#         b = torch.randn((4096, 4096), device=device)
#         # Perform matrix multiplication
#         c = torch.matmul(a, b)
#         # Optionally do something with c to ensure computation isn't optimized away
#         _ = c.mean()  # simple reduction
#         # Let’s sleep just a tiny bit to reduce noise in load
#         time.sleep(0.01)

# def query_gpu_usage():
#     """
#     Helper function to query 'nvidia-smi' and print out GPU usage in real-time.
#     """
#     try:
#         result = subprocess.run(["nvidia-smi", 
#                                  "--query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total",
#                                  "--format=csv,noheader,nounits"],
#                                 capture_output=True, text=True, check=True)
#         print("\n--- GPU USAGE SNAPSHOT ---")
#         print(result.stdout)
#     except FileNotFoundError:
#         print("[ERROR] 'nvidia-smi' not found. Please install NVIDIA drivers correctly.")
#     except subprocess.CalledProcessError as e:
#         print("[ERROR] Failed to run 'nvidia-smi':", e)

# def main():
#     # Number of parallel GPU-heavy processes to spawn. Adjust for your test.
#     num_processes = 1

#     # Start MPS processes
#     processes = []
#     for i in range(num_processes):
#         p = multiprocessing.Process(target=gpu_worker, args=(i,))
#         p.daemon = True
#         p.start()
#         processes.append(p)
    
#     print("[Main] Launched", num_processes, "worker processes.")
#     print("[Main] Monitoring GPU usage (Ctrl+C to stop)...\n")

#     # Continuously monitor GPU usage
#     try:
#         while True:
#             query_gpu_usage()
#             time.sleep(2)  # Adjust frequency as needed
#     except KeyboardInterrupt:
#         print("\n[Main] Stopping all workers...")
#     finally:
#         for p in processes:
#             p.terminate()
#         for p in processes:
#             p.join()
#         print("[Main] Exited cleanly.")

# if __name__ == "__main__":
#     main()


import torch
import time
import os

def benchmark_matrix_multiplication(matrix_size=8192, iterations=10):
    device = torch.device("cuda:0")
    print(f"Process PID: {os.getpid()} | Device: {device} | Matrix size: {matrix_size}x{matrix_size}")
    
    # 创建随机矩阵
    a = torch.randn((matrix_size, matrix_size), device=device)
    b = torch.randn((matrix_size, matrix_size), device=device)

    # 预热 GPU
    torch.matmul(a, b)
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(iterations):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / iterations
    print(f"Average execution time per iteration: {avg_time:.4f} seconds")

if __name__ == "__main__":
    benchmark_matrix_multiplication()
