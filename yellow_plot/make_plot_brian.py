
import numpy as np
import matplotlib.cm as cmap
import brian as b

def plot_2d_input_weights(weights):
    fig = b.figure(0, figsize = (18, 18))
    im2 = b.imshow(weights, interpolation = "nearest", vmin = 0, vmax = 1.0, cmap = cmap.get_cmap('hot_r'))
    b.colorbar(im2)
    fig.canvas.draw()
    return im2, fig

def weights_to_image(weights):
  weights = np.transpose(weights)
  image = np.zeros(shape=(28*20, 28*20))
  for ii in xrange(20):
      for jj in xrange(20):
          for kk in range(28):
              for ll in range(28):
                  y = ii * 28 + kk
                  x = jj * 28 + ll
                  image[x][y] = weights[ii * 20 + jj][kk * 28 + ll]
  image = np.transpose(image)
  return image

w1k = np.load('50k.npy')
w1k_img = weights_to_image(w1k)
plot_2d_input_weights(w1k_img)
b.show()
