import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

x1 = np.genfromtxt('rst_vmem.csv',delimiter=',')
x2 = np.genfromtxt('rst_vo2.csv',delimiter=',')
y = np.genfromtxt('rst_m12.csv',delimiter=',')
y[np.where(y < 0)] = 0 

f = interpolate.interp2d(x1, x2, y, kind='quintic')

print (f(1, 1))

'''
ids[np.where(ids < 0)] = 0 

plt.plot(np.linspace(0, 1, 100), ids)
plt.show()
'''
