
import numpy as np

high = 1.0 / np.sqrt(26)
low = -high
B = np.random.uniform(low=low, high=high, size=(26, 10)) * 2 * 0.12 - 0.12
np.save("B", B)

high = 1.0 / np.sqrt(np.sqrt(26))
low = -high
tmp1 = np.random.uniform(low=low, high=high, size=(26, 1))
tmp2 = np.random.uniform(low=low, high=high, size=(1, 10))
B = np.dot(tmp1, tmp2)
B = B - np.average(B)
np.save("BAD_B", B)
