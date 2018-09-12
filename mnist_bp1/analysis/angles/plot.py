
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
    
good_angles = []
    
B = np.load("../B.npy")
B = B[0:25]
B = np.reshape(B, (-1))

for ii in range(250):
    print (ii)
    
    W2_ii_0 = np.load("../dfa/W2_" + str(ii+1) + "_0.npy")
    W2_ii_0 = W2_ii_0[0:25]
    W2_ii_0 = np.reshape(W2_ii_0, (-1, 1))
    
    W2_ii_1 = np.load("../dfa/W2_" + str(ii+1) + "_1.npy")
    W2_ii_1 = W2_ii_1[0:25]
    W2_ii_1 = np.reshape(W2_ii_1, (-1, 1))
    
    W2_ii_2 = np.load("../dfa/W2_" + str(ii+1) + "_2.npy")
    W2_ii_2 = W2_ii_2[0:25]
    W2_ii_2 = np.reshape(W2_ii_2, (-1, 1))
    
    W2_ii_3 = np.load("../dfa/W2_" + str(ii+1) + "_3.npy")
    W2_ii_3 = W2_ii_3[0:25]
    W2_ii_3 = np.reshape(W2_ii_3, (-1, 1))
    
    angle = angle_between(B, W2_ii_0) * (180.0 / 3.14)
    good_angles.append(angle)
    
    angle = angle_between(B, W2_ii_1) * (180.0 / 3.14)
    good_angles.append(angle)
    
    angle = angle_between(B, W2_ii_2) * (180.0 / 3.14)
    good_angles.append(angle)
    
    angle = angle_between(B, W2_ii_3) * (180.0 / 3.14)
    good_angles.append(angle)
    
#######################################

bad_angles = []
    
B = np.load("../BAD_B.npy")
B = B[0:25]
B = np.reshape(B, (-1))

for ii in range(250):  
    print (ii)

    W2_ii_0 = np.load("../bad_dfa/BAD_W2_" + str(ii+1) + "_0.npy")
    W2_ii_0 = W2_ii_0[0:25]
    W2_ii_0 = np.reshape(W2_ii_0, (-1, 1))
    
    W2_ii_1 = np.load("../bad_dfa/BAD_W2_" + str(ii+1) + "_1.npy")
    W2_ii_1 = W2_ii_1[0:25]
    W2_ii_1 = np.reshape(W2_ii_1, (-1, 1))
    
    W2_ii_2 = np.load("../bad_dfa/BAD_W2_" + str(ii+1) + "_2.npy")
    W2_ii_2 = W2_ii_2[0:25]
    W2_ii_2 = np.reshape(W2_ii_2, (-1, 1))
    
    W2_ii_3 = np.load("../bad_dfa/BAD_W2_" + str(ii+1) + "_3.npy")
    W2_ii_3 = W2_ii_3[0:25]
    W2_ii_3 = np.reshape(W2_ii_3, (-1, 1))
    
    angle = angle_between(B, W2_ii_0) * (180.0 / 3.14)
    bad_angles.append(angle)
    
    angle = angle_between(B, W2_ii_1) * (180.0 / 3.14)
    bad_angles.append(angle)
    
    angle = angle_between(B, W2_ii_2) * (180.0 / 3.14)
    bad_angles.append(angle)
    
    angle = angle_between(B, W2_ii_3) * (180.0 / 3.14)
    bad_angles.append(angle)
    
#######################################

arb_angles = []
    
B = np.load("../B.npy")
B = B[0:25]
B = np.reshape(B, (-1))

for ii in range(250):
    print (ii)
    
    W2_ii_0 = np.load("../weights/W2_" + str(ii+1) + "_0.npy")
    W2_ii_0 = W2_ii_0[0:25]
    W2_ii_0 = np.reshape(W2_ii_0, (-1, 1))
    
    W2_ii_1 = np.load("../weights/W2_" + str(ii+1) + "_1.npy")
    W2_ii_1 = W2_ii_1[0:25]
    W2_ii_1 = np.reshape(W2_ii_1, (-1, 1))
    
    W2_ii_2 = np.load("../weights/W2_" + str(ii+1) + "_2.npy")
    W2_ii_2 = W2_ii_2[0:25]
    W2_ii_2 = np.reshape(W2_ii_2, (-1, 1))
    
    W2_ii_3 = np.load("../weights/W2_" + str(ii+1) + "_3.npy")
    W2_ii_3 = W2_ii_3[0:25]
    W2_ii_3 = np.reshape(W2_ii_3, (-1, 1))
    
    angle = angle_between(B, W2_ii_0) * (180.0 / 3.14)
    arb_angles.append(angle)
    
    angle = angle_between(B, W2_ii_1) * (180.0 / 3.14)
    arb_angles.append(angle)
    
    angle = angle_between(B, W2_ii_2) * (180.0 / 3.14)
    arb_angles.append(angle)
    
    angle = angle_between(B, W2_ii_3) * (180.0 / 3.14)
    arb_angles.append(angle)
    
#######################################

bins = np.linspace(0, 100, 100 * 10)

good_angles = np.reshape(good_angles, (-1))
bad_angles = np.reshape(bad_angles, (-1))
arb_angles = np.reshape(arb_angles, (-1))

plt.rcParams.update({'font.size': 18})
plt.hist(good_angles, alpha=0.5, bins=bins, label="Full Rank FB")
plt.hist(bad_angles, alpha=0.5, bins=bins, label="Rank-1 FB")
plt.hist(arb_angles, alpha=0.5, bins=bins, label="Full Rank FB with BP weights")
plt.xlabel("Angle between FB and Resulting W")
plt.legend(loc='upper right')
plt.show()

    
