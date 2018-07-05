import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

x1 = np.genfromtxt('rst_vmem.csv',delimiter=',')
x2 = np.genfromtxt('rst_vo2.csv',delimiter=',')
y = np.genfromtxt('rst_m12.csv',delimiter=',')
y[np.where(y < 0)] = 0 

print np.min(y)

tck = interpolate.bisplrep(x1, x2, y, kx=5, ky=5)
# print tck

ids = (interpolate.bisplev(1, np.linspace(0, 1, 100), tck))
ids[np.where(ids < 0)] = 0 

plt.plot(np.linspace(0, 1, 100), ids)
plt.show()

'''
x, y = np.mgrid[-1:1:20j, -1:1:20j]
z = (x+y) * np.exp(-6.0*(x*x+y*y))

plt.figure()
plt.pcolor(x, y, z)
plt.colorbar()
plt.title("Sparsely sampled function.")
plt.show()
'''
