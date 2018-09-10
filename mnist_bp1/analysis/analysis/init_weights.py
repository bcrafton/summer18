
import numpy as np
from sklearn.decomposition import PCA

'''
for ii in range(1000):
    W1 = np.random.uniform(size=(785, 25))
    W2 = np.random.uniform(size=(26, 10))
''' 
    
# can just do this ...
W = np.random.uniform(size=(1000, 19885)) * 2 * 0.12 - 0.12

pca = PCA(.95)
pca.fit(W)
print pca.n_components_
