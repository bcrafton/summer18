
import numpy as np
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

w1k = np.load('1k.npy')
w1k_img = weights_to_image(w1k)
plt.imsave('1k.png', w1k_img, cmap=cmap.get_cmap('hot_r'))

w10k = np.load('10k.npy')
w10k_img = weights_to_image(w10k)
plt.imsave('10k.png', w10k_img, cmap=cmap.get_cmap('hot_r'))

w50k = np.load('50k.npy')
w50k_img = weights_to_image(w50k)
plt.imsave('50k.png', w50k_img, cmap=cmap.get_cmap('hot_r'))

w150k = np.load('150k.npy')
w150k_img = weights_to_image(w150k)
plt.imsave('150k.png', w150k_img, cmap=cmap.get_cmap('hot_r'))
