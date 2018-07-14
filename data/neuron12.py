import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
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

##############################################

print "building leak interpolator"

x = leak_vmem
y = leak_m20
m20_fit = interpolate.interp1d(x, y, kind='cubic', fill_value=0.0)

#print sys.getsizeof(m20_fit)

grad = np.gradient(y, x)
m20_vmem_fit = interpolate.interp1d(x, grad, fill_value=0.0)

##############################################

print "building fb interpolator"

x = fb_vmem
y = fb_vo1
vo1_fit = interpolate.interp1d(x, y, kind='cubic', fill_value=0.0)

#print sys.getsizeof(vo1_fit)

grad = np.gradient(y, x)
vo1_vmem_fit = interpolate.interp1d(x, grad, fill_value=0.0)

##############################################

x = fb_vmem
y = fb_m7
m7_fit = interpolate.interp1d(x, y, kind='cubic', fill_value=0.0)

#print sys.getsizeof(m7_fit)

grad = np.gradient(y, x)
m7_vmem_fit = interpolate.interp1d(x, grad, fill_value=0.0)

##############################################

print "building slew interpolator"

x = np.transpose([slew_vo1, slew_vo2])
y = slew_io2
io2_fit = interpolate.LinearNDInterpolator(x, y, fill_value=0.0)

X = np.linspace(0, 1.1, 25)
Y = np.linspace(0, 1.1, 25)

O = np.zeros(shape=(len(X), len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        O[i][j] = io2_fit(X[i], Y[j])
        
grad = np.gradient(O, X, Y, axis=(0, 1))

X1 = np.zeros(shape=(25*25))
Y1 = np.zeros(shape=(25*25))
slew_vo1_grad = np.zeros(shape=(25*25))
slew_vo2_grad = np.zeros(shape=(25*25))

for i in range(len(X)):
    for j in range(len(Y)):
        X1[i * 25 + j] = X[i]
        Y1[i * 25 + j] = Y[j]
        slew_vo1_grad[i * 25 + j] = grad[0][i][j]
        slew_vo2_grad[i * 25 + j] = grad[1][i][j]

io2_vo1_fit = interpolate.LinearNDInterpolator(np.transpose([X1, Y1]), slew_vo1_grad, fill_value=0.0)
io2_vo2_fit = interpolate.LinearNDInterpolator(np.transpose([X1, Y1]), slew_vo2_grad, fill_value=0.0)

##############################################

print "building reset interpolator"

x = np.transpose([rst_vmem, rst_vo2])
y = rst_m12
m12_fit = interpolate.LinearNDInterpolator(x, y, fill_value=0.0)

X = np.linspace(0, 1.1, 25)
Y = np.linspace(0, 1.1, 25)

O = np.zeros(shape=(len(X), len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        O[i][j] = m12_fit(X[i], Y[j])
        
grad = np.gradient(O, X, Y, axis=(0, 1))

X1 = np.zeros(shape=(25*25))
Y1 = np.zeros(shape=(25*25))
rst_mem_grad = np.zeros(shape=(25*25))
rst_vo2_grad = np.zeros(shape=(25*25))

for i in range(len(X)):
    for j in range(len(Y)):
        X1[i * 25 + j] = X[i]
        Y1[i * 25 + j] = Y[j]
        rst_mem_grad[i * 25 + j] = grad[0][i][j]
        rst_vo2_grad[i * 25 + j] = grad[1][i][j]

m12_vmem_fit = interpolate.LinearNDInterpolator(np.transpose([X1, Y1]), rst_mem_grad, fill_value=0.0)
m12_vo2_fit = interpolate.LinearNDInterpolator(np.transpose([X1, Y1]), rst_vo2_grad, fill_value=0.0)

####################################

dt = 1e-6
T = 1e-1
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

def deriv(t, y):
    global iin

    vmem = y[0]
    vo2 = y[1]

    # shud not be doing this ... just collect more points.
    # then we have interpolation issues.
    vmem = min(max(vmem, 0.0), 1.1)    
    vo2 = min(max(vo2, 0.0), 1.1)
    
    vo1 = vo1_fit(vmem)
        
    imem = (iin - m20_fit(vmem) + m7_fit(vmem) - m12_fit(vmem, vo2))
    dvmem_dt = X1 * imem
    
    io2 = io2_fit(vo1, vo2)
    dvo2_dt = X2 * io2
    
    return [dvmem_dt, dvo2_dt]
    
def jac(t, y):
    print "called jac"

    global iin
    
    vmem = y[0]
    vo2 = y[1]
    
    vmem = min(max(vmem, 0.0), 1.1)    
    vo2 = min(max(vo2, 0.0), 1.1)
    
    vo1 = vo1_fit(vmem)
    
    # dimem_dvmem = m7_fit(vmem) - m7_fit(vmem) - m12_fit(vmem, vo2)
    # dimem_dvo2 = -m12_fit(vmem, vo2)
    # dio2_dvmem = io2_fit(vo1, vo2)
    # dio2_dvo2 = io2_fit(vo1, vo2)
    
    dimem_dvmem = m7_vmem_fit(vmem) - m20_vmem_fit(vmem) - m12_vmem_fit(vmem, vo2)
    dimem_dvo2 = -m12_vo2_fit(vmem, vo2)
    # have to do somethng here with partial deriviatie of io2 and vmem
    # it really shudnt be hard
    # we also are not multiplying by (1 / C) and we shud be.
    dio2_dvmem = 0 # probably not right.
    dio2_dvo2 = io2_vo2_fit(vo1, vo2)
    
    return [[dimem_dvmem, dimem_dvo2], [dio2_dvmem, dio2_dvo2]]
    

print "starting sim"
start = time.time()

iin = 0
y0 = [0.0, 0.0]
sol = solve_ivp(deriv, (0.0, 1e-4), y0, method='RK45', jac=jac)

iin = 1e-10
vmem = sol.y[0, -1]
vo2 = sol.y[1, -1]
y0 = [vmem, vo2]
# sol = solve_ivp(deriv, (1e-4, 1e-1), y0, method='BDF', jac=jac)
sol = odeint(deriv, y0, Ts, Dfun=jac, tfirst=True)

end = time.time()
print ("total time taken: " + str(end - start))

#Ts = sol.t
#vmems = sol.y[0, :]
#vo2s = sol.y[1, :]

vmems = sol[:, 0]
vo2s = sol[:, 1]

plt.plot(Ts, vmems)
plt.show()




