
import numpy as np
import matplotlib.pyplot as plt
from gates import *

def square_wave(steps, on_value, off_value, width, period):
    assert(steps % period == 0)
    assert(width < period)
    square = []
    periods = steps / period
    for ii in range(periods):
        off = np.ones(period - width) * off_value
        on = np.ones(width) * on_value
        square.extend(off)
        square.extend(on)
        
    return square

class Memristor:
    def __init__(self):
        self.U = 1e-16
        self.D = 10e-9
        self.W0 = 5e-9
        self.RON = 5e4
        self.ROFF = 1e6
        self.P = 5            
        self.W = self.W0
        
    def step(self, V, dt):
        R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        I = V / R
        F = 1 - (2 * (self.W / self.D) - 1) ** (2 * self.P)
        dwdt = ((self.U * self.RON * I) / self.D) * F
        self.W += dwdt * dt
        return I

class Synapse:
    def __init__(self):
        self.VT = 0.975
        self.C1 = 500e-15
        self.C2 = 250e-15
        
        # these are determined by VT and C1, C2
        self.r1 = 650
        self.r2 = 1300
        
        self.VDD = 1.0
        self.VRESET = 0.32
        self.VTH = 0.5
        # active low!
        self.SPIKE_THRESH = 0.32
        
        # self.PRE_SPIKE = False
        self.LAST_PRE = -1.0
        # self.POST_SPIKE = False
        self.LAST_POST = -1.0
        
        # self.Ts = []
        self.VPREX_OUT = []
        self.VPOSTX_OUT = []
        self.INC_OUT = []
        self.DEC_OUT = []
        self.IM_OUT = []
        
        self.memristor = Memristor()
        
    def step(self, VPRE, VPOST, t, dt):
        # self.Ts.append(t)
    
        if (VPRE <= self.SPIKE_THRESH):
            self.LAST_PRE = t
        if (VPOST <= self.SPIKE_THRESH):
            self.LAST_POST = t
    
        # VPRE'
        VPREN = not_gate(self.VDD, self.VTH, VPRE)
        
        # VPREX
        VPREX = np.clip( self.r1 * (t - self.LAST_PRE), 0, 1.0)
        self.VPREX_OUT.append(VPREX)
        
        # VPOSTX
        VPOSTX = np.clip( self.r2 * (t - self.LAST_POST), 0, 1.0)
        self.VPOSTX_OUT.append(VPOSTX)

        # VPREX nor VPOSTX
        INC = nor_gate(self.VDD, self.VTH, VPREX, VPOSTX)
        self.INC_OUT.append(INC)
        
        # VPREX' nor VPOSTX
        VPREXN = not_gate(self.VDD, self.VTH, VPREX)
        DEC = nor_gate(self.VDD, self.VTH, VPREXN, VPOSTX)
        self.DEC_OUT.append(DEC)
        
        # INC or VPRE'
        READ = or_gate(self.VDD, self.VTH, INC, VPREN)
        
        VP = 0.0
        VM = 0.0
        if (READ == 1.0):
            VP = 0.5
            VM = 0.0
        elif (DEC == 1.0):
            VP = 0.0
            VM = 0.5
            
        IM = self.memristor.step(VP - VM, dt)
        self.IM_OUT.append(IM)
    
        
# sim params
T = 10
dt = 2e-5
steps = int(T / dt) + 1
Ts = np.linspace(0, T, steps)

VPRE_IN = square_wave(steps, 0.32, 1.0, 20, 1000)
VPOST_IN = square_wave(steps, 0.32, 1.0, 20, 1000)

syn = Synapse()
for step in range(steps):
    vpre = VPRE_IN[step] 
    vpost = VPOST_IN[step]
    syn.step(vpre, vpost, step * dt, dt)

plt.subplot(2,2,1)
plt.plot(Ts, syn.VPREX_OUT, Ts, syn.VPOSTX_OUT)

plt.subplot(2,2,2)
plt.plot(Ts, syn.INC_OUT, Ts, syn.DEC_OUT)

plt.subplot(2,2,3)
plt.plot(Ts, syn.IM_OUT)

plt.subplot(2,2,4)
plt.plot(Ts, VPRE_IN, Ts, VPOST_IN)

plt.show()







