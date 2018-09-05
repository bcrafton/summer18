
import numpy as np
import math
import gzip
import time
import pickle
import argparse

from sklearn.decomposition import PCA
#from matplotlib.mlab import PCA

def norm(W1, W2):
    top = np.linalg.norm(W1 - W2) ** 2
    bottom = np.linalg.norm(W1) * np.linalg.norm(W2)
    return top / bottom

NUM_W = 25
Ws = []

for ii in range(NUM_W):
    W1_ii_0 = np.load("W1_" + str(ii+1) + "_0.npy")
    W1_ii_1 = np.load("W1_" + str(ii+1) + "_1.npy")
    W1_ii_2 = np.load("W1_" + str(ii+1) + "_2.npy")
    W1_ii_3 = np.load("W1_" + str(ii+1) + "_3.npy")

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
        
for ii in range(len(Ws)):
    w1 = Ws[0].flatten()
    w2 = Ws[ii].flatten()
    print (norm(w1, w2))
