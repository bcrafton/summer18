import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 18})

B = np.load("B.npy")

bins = np.linspace(-0.25, 0.25, 100)

plt.hist(B, bins=bins)
plt.xlabel("Full Rank FB Dist", fontsize="20")
plt.show()
