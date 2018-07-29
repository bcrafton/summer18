
import numpy as np
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

    
# 784 x 400
weights1 = np.load('XeAe_trained.npy')
# 400 x 784
weights1 = np.transpose(weights1)

weights2 = np.zeros(shape=(28*20, 28*20))

for ii in xrange(20):
    for jj in xrange(20):
        for kk in range(28):
            for ll in range(28):
                y = ii * 28 + kk
                x = jj * 28 + ll
                weights2[x][y] = weights1[ii * 20 + jj][kk * 28 + ll]
                
weights2 = np.transpose(weights2)

# imgplot = plt.imshow(weights2, cmap=cmap.get_cmap('hot_r'))
# plt.show()
plt.imsave('10k.png', weights2, cmap=cmap.get_cmap('hot_r'))
