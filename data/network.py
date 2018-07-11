import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import time
from gates import *
from scipy.integrate import solve_ivp
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

#print sys.getsizeof(m20_fit)

print "building fb interpolator"

x = fb_vmem
y = fb_vo1
vo1_fit = interpolate.interp1d(x, y, kind='cubic', fill_value=0.0)

#print sys.getsizeof(vo1_fit)

x = fb_vmem
y = fb_m7
m7_fit = interpolate.interp1d(x, y, kind='cubic', fill_value=0.0)

#print sys.getsizeof(m7_fit)

print "building slew interpolator"

x = np.transpose([slew_vo1, slew_vo2])
y = slew_io2
io2_fit = interpolate.LinearNDInterpolator(x, y, fill_value=0.0)
# io2_fit = interpolate.NearestNDInterpolator(x, y)

#print sys.getsizeof(io2_fit)

print "building reset interpolator"

x = np.transpose([rst_vmem, rst_vo2])
y = rst_m12
m12_fit = interpolate.LinearNDInterpolator(x, y, fill_value=0.0)
# m12_fit = interpolate.NearestNDInterpolator(x, y)

#print sys.getsizeof(m12_fit)

####################################

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

    vmem_pre = y[0*LAYER1:1*LAYER1]
    vo2_pre =  y[1*LAYER1:2*LAYER1]
    iin_pre =  y[2*LAYER1:3*LAYER1]
    vspk_pre = (vmem_pre > 0.5) * 1.1
    
    dvmem_pre_dt = np.zeros(LAYER1)
    dvo2_pre_dt = np.zeros(LAYER1)
    iin_pre_dt = np.zeros(LAYER1)
    
    OFFSET = 3*LAYER1
    vmem_post = y[OFFSET+0*LAYER2 : OFFSET+1*LAYER2]
    vo2_post =  y[OFFSET+1*LAYER2 : OFFSET+2*LAYER2]
    vspk_post = (vmem_post > 0.5) * 1.1
    
    dvmem_post_dt = np.zeros(LAYER2)
    dvo2_post_dt = np.zeros(LAYER2)

    for i in range(LAYER1):
        vmem_pre[i] = min(max(vmem_pre[i], 0.0), 1.1)    
        vo2_pre[i] = min(max(vo2_pre[i], 0.0), 1.1)
        
        vo1 = vo1_fit(vmem_pre[i])
            
        imem = (iin_pre[i] - m20_fit(vmem_pre[i]) + m7_fit(vmem_pre[i]) - m12_fit(vmem_pre[i], vo2_pre[i]))
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
    
    return np.concatenate( (dvmem_pre_dt, dvo2_pre_dt, iin_pre_dt, dvmem_post_dt, dvo2_post_dt) )


########################
start = time.time()

print "starting sim"
y0 = [0.0] * (3*LAYER1 + 2*LAYER2)
sol = solve_ivp(deriv, (0.0, 1e-4), y0, method='RK45')

print "starting current step"
OFFSET = 3*LAYER1
vmem_pre  = sol.y[0*LAYER1 : 1*LAYER1, -1]
vo2_pre   = sol.y[1*LAYER1 : 2*LAYER1, -1]
vmem_post = sol.y[OFFSET+0*LAYER2 : OFFSET+1*LAYER2, -1]
vo2_post  = sol.y[OFFSET+1*LAYER2 : OFFSET+2*LAYER2, -1]

y0 = np.concatenate( (vmem_pre, vo2_pre, np.ones(LAYER1) * 1e-10, vmem_post, vo2_post) )
sol = solve_ivp(deriv, (1e-4, 1e-1), y0, method='RK45')

end = time.time()
print ("total time taken: " + str(end - start))
########################

Ts = sol.t
vmem_pres = sol.y[0:2, :]
vmem_posts = sol.y[OFFSET:OFFSET+10, :]

plt.plot(Ts, np.transpose(vmem_posts), '.')
# plt.plot(Ts, sol.y[OFFSET, :], Ts, sol.y[OFFSET+10, :])

plt.show()



