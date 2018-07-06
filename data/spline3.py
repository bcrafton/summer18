import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

x1 = np.genfromtxt('rst_vmem.csv',delimiter=',')
x2 = np.genfromtxt('rst_vo2.csv',delimiter=',')
x = np.transpose([x1, x2])
y = np.genfromtxt('rst_m12.csv',delimiter=',')
y[np.where(y < 0)] = 0 

'''
ax = plt.axes(projection='3d')
ax.plot3D(x1, x2, y)
plt.show()
'''

f = interpolate.LinearNDInterpolator(x, y)

'''
vmem = 1
vo2 = np.linspace(0, 1, 100)
m12 = f(vmem, vo2)
plt.plot(vo2, m12)
plt.show()
'''

'''
ids[np.where(ids < 0)] = 0 
plt.plot(np.linspace(0, 1, 100), ids)
plt.show()
'''
