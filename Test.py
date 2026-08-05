import numpy as np
import time

N = 1000

np.random.seed(42)

A = np.random.uniform(-1, 1, (N, N))
B = np.random.uniform(-1, 1, (N, N))

start = time.perf_counter()

C = A @ B

end = time.perf_counter()

print(f"NumPy Time: {end-start:.6f} s")