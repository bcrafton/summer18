
import numpy as np
import matplotlib.pyplot as plt

for sparse in range(1, 10+1):
    ranks = []
    accs = []
    
    for rank in range(sparse, 10+1):
       fname = "sparse" + str(sparse) + "rank" + str(rank) + ".npy"
       acc = np.max(np.load(fname))
       
       print ("sparse", sparse, "rank", rank, acc) 
       
       accs.append(acc)
       ranks.append(rank)
       
    scatter = plt.scatter(ranks, accs, label="Sparse " + str(sparse))
     
plt.xlabel("Rank")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
