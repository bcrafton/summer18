
import numpy as np
import matplotlib.cm as cmap
import brian as b


def plot_2d_input_weights(weights):
    fig = b.figure(0, figsize = (18, 18))
    im2 = b.imshow(weights, vmin = 0, vmax = 1.0, cmap = cmap.get_cmap('hot_r'))
    b.colorbar(im2)
    fig.canvas.draw()
    return im2, fig
    
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

plot_2d_input_weights(weights2)
b.show()
