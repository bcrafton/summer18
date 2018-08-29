
import numpy as np
import math
import gzip
import time
import pickle
import argparse

NUM_W = 25
Ws = []

for ii in range(NUM_W):
    W1_ii_0 = np.load("W1_" + str(ii) + "_0")
    W1_ii_1 = np.load("W1_" + str(ii) + "_1")
    W1_ii_2 = np.load("W1_" + str(ii) + "_2")
    W1_ii_3 = np.load("W1_" + str(ii) + "_3")

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
    
