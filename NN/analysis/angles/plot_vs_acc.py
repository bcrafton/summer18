
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt

#######################################

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#######################################
    
dfa = []
dfa_acc = []
    
B = np.load("../random_feedback/B.npy")
shape = np.shape(B)
if shape[0] < shape[1]:
    B = np.transpose(B)
B = np.reshape(B, (-1))

for ii in range(250):
    print (ii)
    
    W2_ii_0 = np.load("../results/dfa/W2_" + str(ii+1) + "_0.npy")
    W2_ii_0 = W2_ii_0[0:100]
    W2_ii_0 = np.reshape(W2_ii_0, (-1, 1))
    
    angle = angle_between(B, W2_ii_0) * (180.0 / 3.14)
    dfa.append(angle)
    
    acc = np.load("../results/dfa/acc_" + str(ii+1) + "_0.npy")
    dfa_acc.append(np.max(acc))
    
#######################################

dfa1 = []
dfa1_acc = []

B = np.load("../random_feedback/BAD_B.npy")
shape = np.shape(B)
if shape[0] < shape[1]:
    B = np.transpose(B)
B = np.reshape(B, (-1))

for ii in range(250):  
    print (ii)

    W2_ii_0 = np.load("../results/dfa1/W2_" + str(ii+1) + "_1.npy")
    W2_ii_0 = W2_ii_0[0:100]
    W2_ii_0 = np.reshape(W2_ii_0, (-1, 1))
    
    angle = angle_between(B, W2_ii_0) * (180.0 / 3.14)
    dfa1.append(angle)
    
    acc = np.load("../results/dfa1/acc_" + str(ii+1) + "_1.npy")
    dfa1_acc.append(np.max(acc))
        
#######################################
    
sparse_dfa = []
sparse_dfa_acc = []
    
B = np.load("../random_feedback/SPARSE_B.npy")
shape = np.shape(B)
if shape[0] < shape[1]:
    B = np.transpose(B)
B = np.reshape(B, (-1))

for ii in range(250):
    print (ii)
    
    W2_ii_0 = np.load("../results/sparse_dfa/W2_" + str(ii+1) + "_2.npy")
    W2_ii_0 = W2_ii_0[0:100]
    W2_ii_0 = np.reshape(W2_ii_0, (-1, 1))
    
    angle = angle_between(B, W2_ii_0) * (180.0 / 3.14)
    sparse_dfa.append(angle)
    
    acc = np.load("../results/sparse_dfa/acc_" + str(ii+1) + "_2.npy")
    sparse_dfa_acc.append(np.max(acc))
    
#######################################

sparse_dfa1 = []
sparse_dfa1_acc = []
    
B = np.load("../random_feedback/SPARSE_BAD_B.npy")
shape = np.shape(B)
if shape[0] < shape[1]:
    B = np.transpose(B)
B = np.reshape(B, (-1))

for ii in range(250):  
    print (ii)

    W2_ii_0 = np.load("../results/sparse_dfa1/W2_" + str(ii+1) + "_3.npy")
    W2_ii_0 = W2_ii_0[0:100]
    W2_ii_0 = np.reshape(W2_ii_0, (-1, 1))
    
    angle = angle_between(B, W2_ii_0) * (180.0 / 3.14)
    sparse_dfa1.append(angle)
        
    acc = np.load("../results/sparse_dfa1/acc_" + str(ii+1) + "_3.npy")
    sparse_dfa1_acc.append(np.max(acc))
        
#######################################

dfa = np.reshape(dfa, (-1))
dfa1 = np.reshape(dfa1, (-1))
sparse_dfa = np.reshape(sparse_dfa, (-1))
sparse_dfa1 = np.reshape(sparse_dfa1, (-1))

plt.rcParams.update({'font.size': 18})

'''
plt.plot(dfa_acc, dfa, '.')
plt.plot(dfa1_acc, dfa1, '.')
plt.plot(sparse_dfa_acc, sparse_dfa, '.')
plt.plot(sparse_dfa1_acc, sparse_dfa1, '.')
'''

plt.plot(dfa, dfa_acc, '.', label="Full Rank")
plt.plot(dfa1, dfa1_acc, '.', label="Rank-1")
plt.plot(sparse_dfa, sparse_dfa_acc, '.', label="Sparse Full Rank")
plt.plot(sparse_dfa1, sparse_dfa1_acc, '.', label="Sparse Rank-1")

plt.xlabel("Accuracy vs Angle")
plt.legend(loc='upper right')
plt.show()

    
