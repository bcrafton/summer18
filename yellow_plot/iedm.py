
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

w50k = np.load('./data/50k.npy')
w50k_img = weights_to_image(w50k)
plt.imsave('50k.png', w50k_img, cmap=cmap.get_cmap('hot_r'))

w150k = np.load('./data/150k.npy')
w150k_img = weights_to_image(w150k)
plt.imsave('150k.png', w150k_img, cmap=cmap.get_cmap('hot_r'))

for i in range(1, 20+1):
  w = np.load('./data/XeAe_trained_' + str(i*1000) + '.npy')
  w_img = weights_to_image(w)
  plt.imsave(str(i) + 'k.png', w_img, cmap=cmap.get_cmap('hot_r'))
