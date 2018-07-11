import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
import math

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
m20_fit = interpolate.interp1d(x, y, kind='cubic', fill_value=0.0)

print "building fb interpolator"

x = fb_vmem
y = fb_vo1
vo1_fit = interpolate.interp1d(x, y, kind='cubic', fill_value=0.0)

x = fb_vmem
y = fb_m7
m7_fit = interpolate.interp1d(x, y, kind='cubic', fill_value=0.0)

print "building slew interpolator"

x = np.transpose([slew_vo1, slew_vo2])
y = slew_io2
io2_fit = interpolate.LinearNDInterpolator(x, y, fill_value=0.0)
# io2_fit = interpolate.NearestNDInterpolator(x, y)

print "building reset interpolator"

x = np.transpose([rst_vmem, rst_vo2])
y = rst_m12
m12_fit = interpolate.LinearNDInterpolator(x, y, fill_value=0.0)
# m12_fit = interpolate.NearestNDInterpolator(x, y)

####################################

dt = 1e-6
T = 1.0
steps = int(T / dt)
Ts = np.linspace(0, T, steps)

vmem = 0.0
vo2 = 0.0

vmems = np.zeros(steps)
vo2s = np.zeros(steps)
iins = np.zeros(steps)

ico2s = np.zeros(steps)
icmems = np.zeros(steps)

C1 = 500e-15
C2 = 100e-15
X1 = 2e12
X2 = 1e13

########################

print "starting sim"
start = time.time()

y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def deriv(t, y):

    vmem = y[0:2]
    vo2 = y[2:4]
    iin = y[4:6]
    
    dvmem_dt = np.zeros(2)
    dvo2_dt = np.zeros(2)
    iin_dt = np.zeros(2)

    for i in range(2):
        vmem[i] = min(max(vmem[i], 0.0), 1.1)    
        vo2[i] = min(max(vo2[i], 0.0), 1.1)
        
        vo1 = vo1_fit(vmem[i])
            
        imem = (iin[i] - m20_fit(vmem[i]) + m7_fit(vmem[i]) - m12_fit(vmem[i], vo2[i]))
        dvmem_dt[i] = X1 * imem
        
        io2 = io2_fit(vo1, vo2[i])
        dvo2_dt[i] = X2 * io2
    
    return np.concatenate((dvmem_dt, dvo2_dt, iin_dt))

########################

sol = solve_ivp(deriv, (0.0, 1e-4), y0, method='RK45')
y0 = [sol.y[0, -1], sol.y[1, -1], sol.y[2, -1], sol.y[3, -1], 1e-10, 1e-10]

# print y0
print "starting current step"

sol = solve_ivp(deriv, (1e-4, 1e-1), y0, method='RK45')

########################

Ts = sol.t
vmems = sol.y[0, :]
vo2s = sol.y[2, :]

end = time.time()
print ("total time taken: " + str(end - start))

plt.plot(Ts, vmems, Ts, vo2s)
plt.show()




