from mpi4py import MPI
import numpy as np
import time
import csv
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Configurable matrix size
N = 512  # Adjust based on memory & benchmarking goals

# File to log performance metrics
CSV_FILE = "mpi_benchmark_results.csv"
os.makedirs("logs", exist_ok=True)
CSV_PATH = os.path.join("logs", CSV_FILE)

# Root initializes matrices
A, B = None, None
if rank == 0:
    A = np.random.randint(0, 10, (N, N), dtype=np.int32)
    B = np.random.randint(0, 10, (N, N), dtype=np.int32)

# Broadcast B
B = comm.bcast(B if rank == 0 else None, root=0)

# Divide rows across processes
rows_per_proc = N // size
assert N % size == 0, "Matrix size must be divisible by number of processes"

local_A = np.zeros((rows_per_proc, N), dtype=np.int32)
comm.Scatter(A, local_A, root=0)

# Local matrix multiplication and timing
comm.Barrier()  # sync processes
start = time.time()
local_C = np.matmul(local_A, B)
end = time.time()
elapsed_time = end - start

# Gather final result matrix
C = None
if rank == 0:
    C = np.zeros((N, N), dtype=np.int32)
comm.Gather(local_C, C, root=0)

# Save metrics to CSV (only rank 0)
if rank == 0:
    print(f"✅ Done: Matrix {N}x{N}, Processes: {size}, Time: {elapsed_time:.4f} sec")
    headers = ["Matrix Size (N)", "Processes", "Time (s)"]
    new_row = [N, size, round(elapsed_time, 6)]

    # Write or append to CSV
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(new_row)
