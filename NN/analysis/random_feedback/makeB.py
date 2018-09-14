
import numpy as np

high = 1.0 / np.sqrt(np.sqrt(25))
low = -high
tmp1 = np.random.uniform(low=low, high=high, size=(10, 25))
tmp2 = np.random.uniform(low=low, high=high, size=(10, 25))
B = tmp1 * tmp2
B = B - np.average(B)
np.save("B", B)

#########################################################

high = 1.0 / np.sqrt(np.sqrt(25))
low = -high
tmp1 = np.random.uniform(low=low, high=high, size=(10, 1))
tmp2 = np.random.uniform(low=low, high=high, size=(1, 25))
B = np.dot(tmp1, tmp2)
B = B - np.average(B)
np.save("BAD_B", B)

#########################################################

high = 1.0 / np.sqrt(np.sqrt(25))
low = -high

B = np.zeros(shape=(25, 10))
for ii in range(25):
    tmp1 = np.random.uniform(low=low, high=high)
    tmp2 = np.random.uniform(low=low, high=high)
    idx = np.random.randint(low=0, high=10)
    B[ii][idx] = tmp1 * tmp2
B = np.transpose(B)
np.save("SPARSE_B", B)
#print (B)

#########################################################

high = 1.0 / np.sqrt(np.sqrt(25))
low = -high

B = np.zeros(shape=(25, 10))
for ii in range(25):
    tmp1 = np.random.uniform(low=low, high=high)
    tmp2 = np.random.uniform(low=low, high=high)
    idx = 0
    B[ii][0] = tmp1 * tmp2
B = np.transpose(B)
np.save("SPARSE_BAD_B", B)
#print (B)

#########################################################
