
import numpy as np

lo = -1.0 / 3.15
hi = 1.0 / 3.15

tmp1 = np.random.uniform(lo, hi, size=(100, 1))
tmp2 = np.random.uniform(lo, hi, size=(1, 10))
b2 = np.dot(tmp1, tmp2)

mask = np.zeros(shape=(100, 10))
for ii in range(100):
    idx = int(np.random.randint(0, 10))
    mask[ii][idx] = 1.0
    
b2 = b2 * mask

# print (np.linalg.matrix_rank(b2, tol=1e-3))

########

rank = 4

b2 = np.zeros(shape=(100, 10))
for ii in range(rank):
    tmp1 = np.random.uniform(lo, hi, size=(100, 1))
    tmp2 = np.random.uniform(lo, hi, size=(1, 10))
    b2 = b2 + (1.0 / rank) * np.dot(tmp1, tmp2)

mask = np.zeros(shape=(100, 10))
for ii in range(100):
    idx = int(np.random.randint(0, rank))
    mask[ii][idx] = 1.0
    
b2 = b2 * mask

print (np.linalg.matrix_rank(b2, tol=1e-3))
