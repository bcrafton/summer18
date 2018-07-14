import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import time
from gates import *
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy.optimize import approx_fprime
import math
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle

print "loading data"

# leak
leak_vmem = np.genfromtxt('leak_vmem.csv',delimiter=',')
leak_m20 = np.genfromtxt('leak_m20.csv',delimiter=',')

# leak_m20_grad = np.gradient(leak_m20, leak_vmem)

# print leak_m20
# print leak_m20_grad
# plt.plot(leak_vmem, leak_m20, leak_vmem, leak_m20_grad)
# plt.plot(leak_vmem, leak_m20)
# plt.plot(leak_vmem, leak_m20_grad)
# plt.show()

# sf, fb
fb_vmem = np.genfromtxt('fb_vmem.csv',delimiter=',')
fb_vo1 = np.genfromtxt('fb_vo1.csv',delimiter=',')
fb_m7 = np.genfromtxt('fb_m7.csv',delimiter=',')

# fb_vo1_grad = np.gradient(fb_vo1, fb_vmem)
# fb_m7_grad = np.gradient(fb_m7, fb_vmem)

# slew
slew_vo1 = np.genfromtxt('slew_vo1.csv',delimiter=',')
#slew_vo1 = np.reshape(len(slew_vo1), 1)
slew_vo2 = np.genfromtxt('slew_vo2.csv',delimiter=',')
#slew_vo2 = np.reshape(len(slew_vo2), 1)
slew_io2 = np.genfromtxt('slew_io2.csv',delimiter=',')
#slew_io2 = np.reshape(len(slew_io2), 1)

# print np.shape([slew_vo1, slew_vo2])
# print np.shape(np.transpose([slew_vo1, slew_vo2]))
# print np.transpose([slew_vo1, slew_vo2]).tolist()
# print np.array([slew_vo1, slew_vo2]).tolist()
# slew_io2_grad = np.gradient( slew_io2, slew_vo1, slew_vo2, axis=(len(slew_vo1), len(slew_vo2)) )
# slew_io2_grad = np.gradient( slew_io2, slew_vo1, slew_vo2, axis=2 )
# gedit /home/brian/environments/py2/local/lib/python2.7/site-packages/numpy/lib/function_base.py

# oh hold up. we can get all the points we need by just using interpolate. 
# we can probably compute the derivative of interpolation using something online. 

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

##############################################

class MemristorGroup:
    def __init__(self, N, M):
        self.U = 1e-15
        self.D = 10e-9
        self.W0 = 5e-9
        self.RON = 5e4
        self.ROFF = 1e6
        self.P = 5            
        # self.W = self.W0
        self.W = np.random.normal(5e-9, 1e-9, size=(N, M))
        self.W = np.clip(self.W, 3e-9, 8e-9)
        
    def step(self, V):
        R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        I = V / R
        F = 1 - (2 * (self.W / self.D) - 1) ** (2 * self.P)
        dwdt = ((self.U * self.RON * I) / self.D) * F
        # self.W += dwdt * dt
        return I, dwdt

class SynapseGroup:
    def __init__(self, N, M):
        self.N = N
        self.M = M
    
        self.VT = 0.975
        self.C1 = 500e-15
        self.C2 = 250e-15
        
        # these are determined by VT and C1, C2
        self.r1 = 650
        self.r2 = 1300
        
        self.VDD = 1.0
        self.VRESET = 0.32
        self.VTH = 0.5
        self.SPIKE_THRESH = 0.32
        
        self.LAST_PRE = np.ones(N) * -1.0
        self.LAST_POST = np.ones(M) * -1.0
        
        self.memristor = MemristorGroup(N, M)
        
    def step(self, VPRE, VPOST, t):

        pre_spkd = VPRE <= self.SPIKE_THRESH
        pre_nspkd = VPRE > self.SPIKE_THRESH
        self.LAST_PRE = self.LAST_PRE * pre_nspkd
        self.LAST_PRE += pre_spkd * t
            
        post_spkd = VPOST <= self.SPIKE_THRESH
        post_nspkd = VPOST > self.SPIKE_THRESH
        self.LAST_POST = self.LAST_POST * post_nspkd
        self.LAST_POST += post_spkd * t
    
        ########################################################
        # VPRE'
        VPREN = not_gate(self.VDD, self.VTH, VPRE)
        
        # VPREX
        VPREX = np.clip( self.r1 * (t - self.LAST_PRE), 0, 1.0)
        
        # VPOSTX
        VPOSTX = np.clip( self.r2 * (t - self.LAST_POST), 0, 1.0)
        ########################################################
        # VPREX nor VPOSTX
        VPREX = np.array(VPREX)
        VPREX = VPREX.reshape(self.N, 1)
        VPREX = np.repeat(VPREX, self.M, axis=1)

        VPOSTX = np.array(VPOSTX)
        VPOSTX = VPOSTX.reshape(1, self.M)
        VPOSTX = np.repeat(VPOSTX, self.N, axis=0)
        
        INC = nor_gate(self.VDD, self.VTH, VPREX, VPOSTX)
        
        # VPREX' nor VPOSTX
        VPREXN = not_gate(self.VDD, self.VTH, VPREX)
        DEC = nor_gate(self.VDD, self.VTH, VPREXN, VPOSTX)
        
        # INC or VPRE'
        VPREN = VPREN.reshape(self.N, 1)
        READ = or_gate(self.VDD, self.VTH, INC, VPREN)
        ########################################################
            
        VP = (READ == 1.0) * 0.5
        VM = (DEC == 1.0) * 0.5
            
        I, dwdt = self.memristor.step(VP - VM)
        return I, dwdt

####################################

C1 = 500e-15
C2 = 100e-15
X1 = 2e12
X2 = 1e13

NUM_NEURONS = 24 + 48
LAYER1 = 24
LAYER2 = 48

Syn = SynapseGroup(LAYER1, LAYER2)

def deriv(t, y):

    global iin

    vmem_pre = y[0*LAYER1:1*LAYER1]
    vo2_pre =  y[1*LAYER1:2*LAYER1]
    vspk_pre = (vmem_pre > 0.5) * 1.1
    
    dvmem_pre_dt = np.zeros(LAYER1)
    dvo2_pre_dt = np.zeros(LAYER1)
    
    OFFSET = 2*LAYER1
    vmem_post = y[OFFSET+0*LAYER2 : OFFSET+1*LAYER2]
    vo2_post =  y[OFFSET+1*LAYER2 : OFFSET+2*LAYER2]
    vspk_post = (vmem_post > 0.5) * 1.1
    
    dvmem_post_dt = np.zeros(LAYER2)
    dvo2_post_dt = np.zeros(LAYER2)

    for i in range(LAYER1):
        vmem_pre[i] = min(max(vmem_pre[i], 0.0), 1.1)    
        vo2_pre[i] = min(max(vo2_pre[i], 0.0), 1.1)
        
        vo1 = vo1_fit(vmem_pre[i])
            
        imem = (iin - m20_fit(vmem_pre[i]) + m7_fit(vmem_pre[i]) - m12_fit(vmem_pre[i], vo2_pre[i]))
        dvmem_pre_dt[i] = X1 * imem
        
        io2 = io2_fit(vo1, vo2_pre[i])
        dvo2_pre_dt[i] = X2 * io2
        
    # HUGE WIN: We only need M + N timer circuits. NOT M * N!!!
    # This could be huge for a digital design with a large number of timer circuits.
    
    '''
    vspk_pre = np.array(vspk_pre)
    vspk_pre = vspk_pre.reshape(LAYER1, 1)
    vspk_pre = np.repeat(vspk_pre, LAYER2, axis=1)
    
    vspk_post = np.array(vspk_post)
    vspk_post = vspk_post.reshape(1, LAYER2)
    vspk_post = np.repeat(vspk_post, LAYER1, axis=0)
    '''
    
    Isyn, dwdt = Syn.step(vspk_pre, vspk_post, t)
    Isyn = np.sum(Isyn, axis=0)
    Isyn = Isyn / 2.5e5
    
    for i in range(LAYER2):
        vmem_post[i] = min(max(vmem_post[i], 0.0), 1.1)    
        vo2_post[i] = min(max(vo2_post[i], 0.0), 1.1)
        
        vo1 = vo1_fit(vmem_post[i])
            
        imem = (Isyn[i] - m20_fit(vmem_post[i]) + m7_fit(vmem_post[i]) - m12_fit(vmem_post[i], vo2_post[i]))
        dvmem_post_dt[i] = X1 * imem
        
        io2 = io2_fit(vo1, vo2_post[i])
        dvo2_post_dt[i] = X2 * io2
    
    return np.concatenate( (dvmem_pre_dt, dvo2_pre_dt, dvmem_post_dt, dvo2_post_dt) )

def deriv2(t, y):

    # assert(False)

    global iin

    vmem_pre = y[0*LAYER1:1*LAYER1]
    vo2_pre =  y[1*LAYER1:2*LAYER1]
    vspk_pre = (vmem_pre > 0.5) * 1.1
    
    d2vmem_pre_dt = np.zeros(LAYER1)
    d2vo2_pre_dt = np.zeros(LAYER1)
    
    OFFSET = 2*LAYER1
    vmem_post = y[OFFSET+0*LAYER2 : OFFSET+1*LAYER2]
    vo2_post =  y[OFFSET+1*LAYER2 : OFFSET+2*LAYER2]
    vspk_post = (vmem_post > 0.5) * 1.1
    
    d2vmem_post_dt = np.zeros(LAYER2)
    d2vo2_post_dt = np.zeros(LAYER2)

    for i in range(LAYER1):
        vmem_pre[i] = min(max(vmem_pre[i], 0.0), 1.1)    
        vo2_pre[i] = min(max(vo2_pre[i], 0.0), 1.1)
        
        vo1 = vo1_fit(vmem_pre[i])
            
        imem = (iin - m20_fit(vmem_pre[i]) + m7_fit(vmem_pre[i]) - m12_fit(vmem_pre[i], vo2_pre[i]))
        dvmem_pre_dt = X1 * imem
        
        io2 = io2_fit(vo1, vo2_pre[i])
        dvo2_pre_dt = X2 * io2
        
        ### compute 2nd derivative.
        
        d2vmem_pre_dt[i] = dvmem_pre_dt * (m7_vmem_fit(vmem_pre[i]) - m20_vmem_fit(vmem_pre[i]) - m12_vmem_fit(vmem_pre[i], vo2_pre[i])) + dvo2_pre_dt * m12_vo2_fit(vmem_pre[i], vo2_pre[i])
        # this part is incorrect! we are doing 'dvmem_pre_dt * io2_vo1_fit(vo1)' instead of using dvo1 because we dont have it
        d2vo2_pre_dt[i] = dvmem_pre_dt * io2_vo1_fit(vo1, vo2_pre[i]) + dvo2_pre_dt * io2_vo2_fit(vo1, vo2_pre[i])
    
    Isyn, dwdt = Syn.step(vspk_pre, vspk_post, t)
    Isyn = np.sum(Isyn, axis=0)
    Isyn = Isyn / 2.5e5
    
    for i in range(LAYER2):
        vmem_post[i] = min(max(vmem_post[i], 0.0), 1.1)    
        vo2_post[i] = min(max(vo2_post[i], 0.0), 1.1)
        
        vo1 = vo1_fit(vmem_post[i])
            
        imem = (Isyn[i] - m20_fit(vmem_post[i]) + m7_fit(vmem_post[i]) - m12_fit(vmem_post[i], vo2_post[i]))
        dvmem_post_dt = X1 * imem
        
        io2 = io2_fit(vo1, vo2_post[i])
        dvo2_post_dt = X2 * io2
        
        ### compute 2nd derivative.
        
        d2vmem_post_dt[i] = dvmem_post_dt * (m7_vmem_fit(vmem_post[i]) - m20_vmem_fit(vmem_post[i]) - m12_vmem_fit(vmem_post[i], vo2_post[i])) + dvo2_post_dt * m12_vo2_fit(vmem_post[i], vo2_post[i])
        # this part is incorrect! we are doing 'dvmem_post_dt * io2_vo1_fit(vo1)' instead of using dvo1 because we dont have it
        d2vo2_post_dt[i] = dvmem_post_dt * io2_vo1_fit(vo1, vo2_post[i]) + dvo2_post_dt * io2_vo2_fit(vo1, vo2_post[i])
    
    return np.concatenate( (d2vmem_pre_dt, d2vo2_pre_dt, d2vmem_post_dt, d2vo2_post_dt) )

########################

iin = 0

dt = 1e-6
T = 1e-4
steps = int(T / dt)
Ts = np.linspace(0, T, steps)

y0 = [0.0] * (2*LAYER1 + 2*LAYER2)
sol = solve_ivp(deriv, (0.0, 1e-4), y0, method='RK45')
# sol, info = odeint(deriv, y0, Ts, args=(0,), Dfun=deriv2, tfirst=True, full_output=1)
# print info

########################

iin = 1e-10

dt = 1e-6
T = 1e-1
steps = int(T / dt)
Ts = np.linspace(0, T, steps)

start = time.time()
print "starting sim"

OFFSET = 2*LAYER1
vmem_pre  = sol.y[0*LAYER1 : 1*LAYER1, -1]
vo2_pre   = sol.y[1*LAYER1 : 2*LAYER1, -1]
vmem_post = sol.y[OFFSET+0*LAYER2 : OFFSET+1*LAYER2, -1]
vo2_post  = sol.y[OFFSET+1*LAYER2 : OFFSET+2*LAYER2, -1]
'''
vmem_pre  = sol[-1, 0*LAYER1 : 1*LAYER1]
vo2_pre   = sol[-1, 1*LAYER1 : 2*LAYER1]
vmem_post = sol[-1, OFFSET+0*LAYER2 : OFFSET+1*LAYER2]
vo2_post  = sol[-1, OFFSET+1*LAYER2 : OFFSET+2*LAYER2]
'''

# y0 = np.concatenate( (vmem_pre, vo2_pre, vmem_post, vo2_post) )
sol = solve_ivp(deriv, (1e-4, 1e-1), y0, method='BDF', jac=deriv2)
# sol, info = odeint(deriv, y0, Ts, args=(1e-10,), Dfun=deriv2, full_output=1)
# print info['mused']

eq = ode(deriv, deriv2).set_integrator('vode', method='bdf')
eq.set_initial_value(y0, 0.0).set_f_params(1e-10).set_jac_params(1e-10)
while eq.successful() and eq.t < 1e-1:
    print(eq.t+dt, eq.integrate(eq.t+dt))

end = time.time()
print ("total time taken: " + str(end - start))
########################

vmem_posts = sol[:, OFFSET:OFFSET+10]

plt.plot(Ts, vmem_posts, '.')
# plt.plot(Ts, sol.y[OFFSET, :], Ts, sol.y[OFFSET+10, :])

plt.show()



