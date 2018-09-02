
import numpy as np
import math
import gzip
import time
import pickle
import argparse

from sklearn.decomposition import PCA
# from matplotlib.mlab import PCA

NUM_W = 250
Ws = []

for ii in range(NUM_W):
    W1_ii_0 = np.load("./weights/W1_" + str(ii+1) + "_0.npy")
    W1_ii_1 = np.load("./weights/W1_" + str(ii+1) + "_1.npy")
    W1_ii_2 = np.load("./weights/W1_" + str(ii+1) + "_2.npy")
    W1_ii_3 = np.load("./weights/W1_" + str(ii+1) + "_3.npy")

    Ws.append(W1_ii_0)
    Ws.append(W1_ii_1)
    Ws.append(W1_ii_2)
    Ws.append(W1_ii_3)

for ii in range(len(Ws)):
    w = Ws[ii]
    w = w.reshape(-1, 1)
    
    if ii==0:
        mat = w
    else:
        mat = np.concatenate((mat, w), axis=1)

print (np.shape(mat))   

# sklearn
pca = PCA(.95)

pca.fit(mat)
print pca.n_components_

pca.fit(np.transpose(mat))
print pca.n_components_

# matplotlib
#pca = PCA(mat)

vals, vecs = np.linalg.eig( np.dot(np.transpose(mat), mat) )
print (vals)
