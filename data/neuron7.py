
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from scipy import interpolate
from scipy.optimize import curve_fit

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

def poly55(X, p00, p10, p01, p20, p11, p02, p30, p21, p12, p03, p40, p31, p22, p13, p04, p50, p41, p32, p23, p14, p05):

    # print np.shape(X)

    if len(np.shape(X)) > 1:
        x = X[:, 0]
        y = X[:, 1]
    else:
        x = X[0]
        y = X[1]
        
    return p00 + p10*x + p01*y + p20*x**2 + p11*x*y + p02*y**2 + p30*x**3 + p21*x**2*y + p12*x*y**2 + p03*y**3 + p40*x**4 + p31*x**3*y + p22*x**2*y**2 + p13*x*y**3 + p04*y**4 + p50*x**5 + p41*x**4*y + p32*x**3*y**2 + p23*x**2*y**3 + p14*x*y**4 + p05*y**5

popt, pcov = curve_fit(poly55, np.transpose([slew_vo1, slew_vo2]), slew_io2)
# plt.plot(slew_vo1, poly55(np.transpose([slew_vo1, slew_vo2]), *popt))
# plt.show()
    
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

y0 = [0, 0, 0]

def deriv(t, y):
    vmem = y[0]
    vo2 = y[1]
    iin = y[2]

    # shud not be doing this ... just collect more points.
    vmem = min(max(vmem, 0.0), 1.0)    
    vo2 = min(max(vo2, 0.0), 1.0)
    
    vo1 = vo1_fit(vmem)
        
    imem = (iin - m20_fit(vmem) + m7_fit(vmem) - m12_fit(vmem, vo2))
    dvmem_dt = (1 / C1) * imem
    
    # io2 = io2_fit(vo1, vo2)
    io2 = max(poly55(np.transpose([vo1, vo2]), *popt), 1e-12)
    dvo2_dt = (1 / C2) * io2
    
    return [dvmem_dt, dvo2_dt, 0.0]

'''
for i in range(steps):
    
    t = Ts[i]

    if (t > 1e-4):
        iin = 1e-10  
    else:
        iin = 0
        
    vo1 = vo1_fit(vmem)
    
    io2 = io2_fit(vo1, vo2)
    dvdt = (1 / C2) * io2
    vo2 = vo2 + dvdt * dt
    vo2 = min(max(vo2, 0.0), 1.0)
        
    icmem = (iin - m20_fit(vmem) + m7_fit(vmem) - m12_fit(vmem, vo2))
    dvdt = (1 / C1) * icmem
    vmem = vmem + dvdt * dt
    vmem = min(max(vmem, 0.0), 1.0)
    
    vmems[i] = vmem
    vo2s[i] = vo2
'''

sol = solve_ivp(deriv, (0, 1e-4), y0, method='RK45')
vmem = sol.y[0, -1]
vo2 = sol.y[1, -1]
y0 = [vmem, vo2, 1e-10]
sol = solve_ivp(deriv, (1e-4, 1e-1), y0, method='RK45')

Ts = sol.t
vmems = sol.y[0, :]
vo2s = sol.y[1, :]

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

# plt.subplot(2,2,1)
# plt.plot(Ts, ico2s, Ts, icmems)

# plt.subplot(2,2,2)
plt.plot(Ts, vmems, Ts, vo2s)

# plt.subplot(2,2,3)

# plt.subplot(2,2,4)

plt.show()


#######################




