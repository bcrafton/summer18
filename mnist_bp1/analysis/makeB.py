
import numpy as np

high = 1.0 / np.sqrt(np.sqrt(26))
low = -high
tmp1 = np.random.uniform(low=low, high=high, size=(26, 10))
tmp2 = np.random.uniform(low=low, high=high, size=(26, 10))
B = tmp1 * tmp2
B = B - np.average(B)
#np.save("B", B)

#########################################################

high = 1.0 / np.sqrt(np.sqrt(26))
low = -high
tmp1 = np.random.uniform(low=low, high=high, size=(26, 1))
tmp2 = np.random.uniform(low=low, high=high, size=(1, 10))
B = np.dot(tmp1, tmp2)
B = B - np.average(B)
#np.save("BAD_B", B)

#########################################################

high = 1.0 / np.sqrt(np.sqrt(26))
low = -high

B = np.zeros(shape=(26, 10))
for ii in range(26):
    tmp1 = np.random.uniform(low=low, high=high)
    tmp2 = np.random.uniform(low=low, high=high)
    idx = np.random.randint(low=0, high=10)
    B[ii][idx] = tmp1 * tmp2
np.save("SPARSE_B", B)
print (B)

#########################################################

high = 1.0 / np.sqrt(np.sqrt(26))
low = -high

B = np.zeros(shape=(26, 10))
for ii in range(26):
    tmp1 = np.random.uniform(low=low, high=high)
    tmp2 = np.random.uniform(low=low, high=high)
    idx = 0
    B[ii][0] = tmp1 * tmp2
np.save("SPARSE_BAD_B", B)

print (B)

#########################################################
