import numpy as np

NUM = 2
ws = []
for ii in range(NUM):
    w1 = np.load("W1_" + str(ii) + "_0.npy")
    ws.append(w1)
    
for ii in range(NUM):
    print (ws[ii])
