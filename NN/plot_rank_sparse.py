
import numpy as np
import matplotlib.pyplot as plt

for sparse in range(1, 10+1):
    ranks = []
    accs = []
    
    for rank in range(sparse, 10+1):
       fname = "sparse" + str(sparse) + "rank" + str(rank) + ".npy"
       acc = np.max(np.load(fname))
       
       accs.append(acc)
       ranks.append(rank)
       
    scatter = plt.scatter(rank, acc)
     
plt.show()
