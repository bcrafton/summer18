import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import time

try:
    import cPickle as pickle
except ImportError:
    import pickle

print "loading interpolator"
with open('io2_func.pkl', 'rb') as f:
    io2_func = pickle.load(f)
with open('imem_func.pkl', 'rb') as f:
    imem_func = pickle.load(f)

class Neuron:

    def __init__(self, io2_func, imem_func):
        self.io2_func = io2_func
        self.imem_func = imem_func
        self.C1 = 500e-15
        self.C2 = 100e-15
        
        self.vmem = 0
        self.vo2 = 0
        
        self.vmems = []
        self.vo2s = []
        self.icmems = []
        self.ico2s = []
        
    def step(self, i_in, dt):
        icmem = self.imem_func(self.vmem, self.vo2, i_in)
        dvdt = (1 / self.C1) * icmem
        self.vmem = self.vmem + dvdt * dt
        
        ico2 = self.io2_func(self.vmem, self.vo2, i_in)
        dvdt = (1 / self.C2) * ico2
        self.vo2 = self.vo2 + dvdt * dt
        
        self.vmems.append(self.vmem)
        self.vo2s.append(self.vo2)
        self.icmems.append(icmem)
        self.ico2s.append(ico2)

####################################

dt = 1e-6
T = 1.0
steps = int(T / dt)
Ts = np.linspace(0, T, steps)

N = Neuron(io2_func, imem_func)

########################

print "starting sim"
start = time.time()

for i in range(steps):
    t = Ts[i]
    if (t > 1e-4):
        i_in = 1e-9    
    else:
        i_in = 0

    N.step(i_in, dt)
    
end = time.time()
print ("total time taken:" + str(end - start))
#######################

plt.subplot(2,2,1)
plt.plot(Ts, N.ico2s, Ts, N.icmems)

plt.subplot(2,2,2)
plt.plot(Ts, N.vmems, Ts, N.vo2s)

# plt.subplot(2,2,3)

# plt.subplot(2,2,4)

plt.show()


#######################




