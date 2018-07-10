import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import time

try:
    import cPickle as pickle
except ImportError:
    import pickle

print "loading data"

# leak
leak_vmem = np.genfromtxt('leak_vmem.csv',delimiter=',')
leak_m20 = np.genfromtxt('leak_m20.csv',delimiter=',')

# sf, fb
fb_vmem = np.genfromtxt('fb_vmem.csv',delimiter=',')
fb_vo1 = np.genfromtxt('fb_vo1.csv',delimiter=',')
fb_m7 = np.genfromtxt('fb_m7.csv',delimiter=',')

# slew
slew_vo1 = np.genfromtxt('slew_vo1.csv',delimiter=',')
slew_vo2 = np.genfromtxt('slew_vo2.csv',delimiter=',')
slew_io2 = np.genfromtxt('slew_io2.csv',delimiter=',')

# reset
rst_vmem = np.genfromtxt('rst_vmem.csv',delimiter=',')
rst_vo2 = np.genfromtxt('rst_vo2.csv',delimiter=',')
rst_m12 = np.genfromtxt('rst_m12.csv',delimiter=',')

print "building leak interpolator"

x = leak_vmem
y = leak_m20
m20_fit = interpolate.interp1d(x, y, kind='cubic')

print "building fb interpolator"

x = fb_vmem
y = fb_vo1
vo1_fit = interpolate.interp1d(x, y, kind='cubic')

x = fb_vmem
y = fb_m7
m7_fit = interpolate.interp1d(x, y, kind='cubic')

print "building slew interpolator"

x = np.transpose([slew_vo1, slew_vo2])
y = slew_io2
io2_fit = interpolate.LinearNDInterpolator(x, y, fill_value=0.0)

print "building reset interpolator"

x = np.transpose([rst_vmem, rst_vo2])
y = rst_m12
m12_fit = interpolate.LinearNDInterpolator(x, y, fill_value=0.0)

####################################

dt = 1e-6
T = 1.0
steps = int(T / dt)
Ts = np.linspace(0, T, steps)

vmem = 0
vo2 = 0

vmems = np.zeros(steps)
vo2s = np.zeros(steps)
iins = np.zeros(steps)

ico2s = np.zeros(steps)
icmems = np.zeros(steps)

C1 = 500e-15
C2 = 100e-15

########################

print "starting sim"
start = time.time()

for i in range(steps):
    
    t = Ts[i]

    if (t > 1e-4):
        iin = 1e-9  
    else:
        iin = 0
        
    vo1 = vo1_fit(vmem)
    
    dvdt = (1 / C2) * io2_fit(vo1, vo2)
    vo2 = vo2 + dvdt * dt
    vo2 = min(max(vo2, 0.0), 1.0)
        
    icmem = (iin - m20_fit(vmem) + m7_fit(vmem) - m12_fit(vmem, vo2))
    dvdt = (1 / C1) * icmem
    vmem = vmem + dvdt * dt
    vmem = min(max(vmem, 0.0), 1.0)
    
    vmems[i] = vmem
    vo2s[i] = vo2
    
end = time.time()
print ("total time taken: " + str(end - start))

#######################
'''
with open('io2_fit.pkl', 'wb') as f:
    pickle.dump(io2_fit, f)
    
with open('imem_fit.pkl', 'wb') as f:
    pickle.dump(imem_fit, f)
''' 
#######################
'''    
with open('io2_fit.pkl', 'rb') as f:
    io2_fit = pickle.load(f)
    
with open('imem_fit.pkl', 'rb') as f:
    imem_fit = pickle.load(f)
'''
#######################

plt.subplot(2,2,1)
plt.plot(Ts, ico2s, Ts, icmems)

plt.subplot(2,2,2)
plt.plot(Ts, vmems, Ts, vo2s)

# plt.subplot(2,2,3)

# plt.subplot(2,2,4)

plt.show()


#######################




