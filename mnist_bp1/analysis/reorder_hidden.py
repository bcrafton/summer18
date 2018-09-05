import numpy as np

W1_1 = np.load("W1_1.npy")
W1_1 = np.reshape(W1_1, (785, 100))

W2_1 = np.load("W2_1.npy")
W2_1 = np.reshape(W2_1, (101, 10))

#####################################

W1_2 = np.load("W1_2.npy")
W1_2 = np.reshape(W1_2, (785, 100))

W2_2 = np.load("W2_2.npy")
W2_2 = np.reshape(W2_2, (101, 10))

#####################################

order = np.argsort(np.transpose(W2_1)[0][:-1])

################################################

tmp1 = np.copy(W2_1[order[0]])
W2_1[0:100] = W2_1[order]
tmp2 = np.copy(W2_1[0])
assert(np.all(tmp1 == tmp2))

W1_1 = np.transpose(W1_1)
tmp1 = np.copy(W1_1[order[0]])
W1_1 = W1_1[order]
tmp2 = np.copy(W1_1[0])
assert(np.all(tmp1 == tmp2))
W1_1 = np.transpose(W1_1)


