
import numpy as np

B = np.random.uniform(low=0.0, high=1.0, size=(26, 10)) * 2 * 0.12 - 0.12
print np.average(B)

tmp1 = np.random.uniform(low=0.0, high=1.0, size=(26, 1))
tmp2 = np.random.uniform(low=0.0, high=1.0, size=(1, 10))
B = np.dot(tmp1, tmp2)
B = B - np.average(B)
print np.average(B)

np.save("BAD_B", B)
