
import numpy as np
import math
import gzip
import time
import pickle
import argparse

from sklearn.decomposition import PCA
# from matplotlib.mlab import PCA

NUM_W = 450
W1s = []
W2s = []

def reorder(W1, W2):
    order = np.argsort(np.transpose(W2)[0][:-1])

    tmp1 = np.copy(W2[order[0]])
    W2[0:25] = W2[order]
    tmp2 = np.copy(W2[0])
    assert(np.all(tmp1 == tmp2))

    W1 = np.transpose(W1)
    tmp1 = np.copy(W1[order[0]])
    W1 = W1[order]
    tmp2 = np.copy(W1[0])
    assert(np.all(tmp1 == tmp2))
    W1 = np.transpose(W1)

    return W1, W2

'''
def reorder(W1_ref, W1, W2_ref, W2):
    # print "-------------"

    error = np.zeros(shape=(25, 25))
    
    W1_ref = np.transpose(W1_ref)
    W1 = np.transpose(W1)
    
    for ii in range(25):
        for jj in range(25):
            error[ii][jj] += np.average((W1_ref[ii] - W1[jj]) ** 2)

    for ii in range(25):
        for jj in range(25):
            error[ii][jj] += np.average((W2_ref[ii] - W2[jj]) ** 2)

    placed = set()
    order = [0] * 25
    for ii in range(25):
        argsort = np.argsort(error[ii])
        for jj in range(25):
            argmin = argsort[jj]
            if argmin not in placed:
                order[ii] = argmin
                placed.add(argmin)
                # print "picked: " + str(argmin)
                break
            else:
                # print "argmin already used! " + str(argmin) + " " + str(argsort) + " " + str(placed)
                pass
                
                
    # print order
    
    W2[0:25] = W2[order]
    W1 = W1[order]
            
    W1 = np.transpose(W1)
    return W1, W2
    
'''
'''
def reorder(W1, W2):
    total = np.zeros(25)
    
    for ii in range(10):
        predict = np.zeros(10)
        predict[ii] = 1.0
        
        total += np.dot(W2, predict)[0:25]
        
    order = np.argsort(np.transpose(total))
    
    W2[0:25] = W2[order]
        
    W1 = np.transpose(W1)
    W1 = W1[order]
    W1 = np.transpose(W1)
        
    return W1, W2
'''     

W1_ref = W1_ii_0 = np.load("../weights/W1_1_0.npy")
W2_ref = W2_ii_0 = np.load("../weights/W2_1_0.npy")

for ii in range(NUM_W):
    print (ii)

    W1_ii_0 = np.load("../weights/W1_" + str(ii+1) + "_0.npy")
    W1_ii_0 = W1_ii_0 - np.average(W1_ii_0)
    
    W1_ii_1 = np.load("../weights/W1_" + str(ii+1) + "_1.npy")
    W1_ii_1 = W1_ii_1 - np.average(W1_ii_1)
    
    W1_ii_2 = np.load("../weights/W1_" + str(ii+1) + "_2.npy")
    W1_ii_2 = W1_ii_2 - np.average(W1_ii_2)
    
    W1_ii_3 = np.load("../weights/W1_" + str(ii+1) + "_3.npy")    
    W1_ii_3 = W1_ii_3 - np.average(W1_ii_3)

    W2_ii_0 = np.load("../weights/W2_" + str(ii+1) + "_0.npy")
    W2_ii_0 = W2_ii_0 - np.average(W2_ii_0)
    
    W2_ii_1 = np.load("../weights/W2_" + str(ii+1) + "_1.npy")
    W2_ii_1 = W2_ii_1 - np.average(W2_ii_1)
    
    W2_ii_2 = np.load("../weights/W2_" + str(ii+1) + "_2.npy")
    W2_ii_2 = W2_ii_2 - np.average(W2_ii_2)
    
    W2_ii_3 = np.load("../weights/W2_" + str(ii+1) + "_3.npy")
    W2_ii_3 = W2_ii_3 - np.average(W2_ii_3)
    
    '''
    W1_ii_0, W2_ii_0 = reorder(W1_ref, W1_ii_0, W2_ref, W2_ii_0)
    W1_ii_1, W2_ii_1 = reorder(W1_ref, W1_ii_1, W2_ref, W2_ii_1)
    W1_ii_2, W2_ii_2 = reorder(W1_ref, W1_ii_2, W2_ref, W2_ii_2)
    W1_ii_3, W2_ii_3 = reorder(W1_ref, W1_ii_3, W2_ref, W2_ii_3)
    '''
    
    W1_ii_0, W2_ii_0 = reorder(W1_ii_0, W2_ii_0)
    W1_ii_1, W2_ii_1 = reorder(W1_ii_1, W2_ii_1)
    W1_ii_2, W2_ii_2 = reorder(W1_ii_2, W2_ii_2)
    W1_ii_3, W2_ii_3 = reorder(W1_ii_3, W2_ii_3)
    
    W1s.append(W1_ii_0)
    W1s.append(W1_ii_1)
    W1s.append(W1_ii_2)
    W1s.append(W1_ii_3)
    
    W2s.append(W2_ii_0)
    W2s.append(W2_ii_1)
    W2s.append(W2_ii_2)
    W2s.append(W2_ii_3)

for ii in range(len(W1s)):
    print (ii)

    w1 = W1s[ii]
    w1 = w1.reshape(-1, 1)
    
    w2 = W2s[ii]
    w2 = w2.reshape(-1, 1)
    
    # w = np.concatenate((w1, w2), axis=0)
    w = w2
    
    if ii==0:
        mat = w
    else:
        mat = np.concatenate((mat, w), axis=1)

print (np.shape(mat))

# X : array-like, shape (n_samples, n_features)
# Training data, where n_samples in the number of samples and n_features is the number of features.

# sklearn
pca = PCA(.95)

# pca.fit(mat)
# print pca.n_components_

mat = np.transpose(mat)
# mean, std = np.average(mat), np.std(mat)
# mat = np.random.normal(loc=mean, scale=std, size=(np.shape(mat)))

pca.fit(mat)
print pca.n_components_

# matplotlib
#pca = PCA(mat)

#vals, vecs = np.linalg.eig( np.dot(np.transpose(mat), mat) )
#print (vals)
