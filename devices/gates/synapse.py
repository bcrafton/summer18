
import numpy as np
import matplotlib.pyplot as plt
from gates import *
import time

def square_wave(steps, on_value, off_value, delay, width, period):
    assert(steps % period == 0)
    assert((width + delay) < period)
    square = []
    periods = steps / period
    for ii in range(periods):
        if (delay > 0):
            off = np.ones(delay) * off_value
            square.extend(off)

        on = np.ones(width) * on_value
        square.extend(on)

        off = np.ones(period - delay - width) * off_value
        square.extend(off)
        
    return square

class Memristor:
    def __init__(self):
        self.U = 1e-15
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
T = 20
dt = 2e-5
steps = int(T / dt) + 1
Ts = np.linspace(0, T, steps)

VPRE_IN1 = square_wave(steps, 0.32, 1.0, 100, 20, 1000)
VPRE_IN2 = square_wave(steps, 0.32, 1.0, 375, 20, 1000)
VPRE_IN3 = square_wave(steps, 0.32, 1.0, 500, 20, 1000)
VPRE_IN4 = square_wave(steps, 0.32, 1.0, 700, 20, 1000)

VPOST_IN = square_wave(steps, 0.32, 1.0, 400, 20, 1000)

syn1 = Synapse()
syn2 = Synapse()
syn3 = Synapse()
syn4 = Synapse()

# start timer
start = time.time()

for step in range(steps):
    vpre1 = VPRE_IN1[step] 
    vpre2 = VPRE_IN2[step] 
    vpre3 = VPRE_IN3[step] 
    vpre4 = VPRE_IN4[step] 

    vpost = VPOST_IN[step]

    syn1.step(vpre1, vpost, step * dt, dt)
    syn2.step(vpre2, vpost, step * dt, dt)
    syn3.step(vpre3, vpost, step * dt, dt)
    syn4.step(vpre4, vpost, step * dt, dt)

# end timer
end = time.time()
print ("total time taken:" + str(end - start))

plot_syn = syn2

plt.subplot(2,2,1)
# plt.plot(Ts, plot_syn.VPREX_OUT, Ts, plot_syn.VPOSTX_OUT)

plt.subplot(2,2,2)
# plt.plot(Ts, plot_syn.INC_OUT, Ts, plot_syn.DEC_OUT)

plt.subplot(2,2,3)
plt.plot(Ts, plot_syn.IM_OUT)

plt.subplot(2,2,4)
plt.plot(Ts, VPRE_IN1, Ts, VPOST_IN)

plt.show()







