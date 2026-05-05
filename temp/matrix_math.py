import numpy as np

n = 100
m = 12

A = np.ones((n, m), dtype=int)

print(np.sum(A, axis=1))
