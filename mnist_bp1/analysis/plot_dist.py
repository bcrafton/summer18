import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

samples = np.load("B.npy")

count, bins, ignored = plt.hist(samples, bins=200)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()
