import numpy as np
import time

N = 512
A = np.random.randint(0, 10, (N, N), dtype=np.int32)
B = np.random.randint(0, 10, (N, N), dtype=np.int32)

start = time.time()
C = np.matmul(A, B)
end = time.time()

print(f"✅ Serial Execution: Matrix {N}x{N}, Time: {end - start:.4f} sec")
