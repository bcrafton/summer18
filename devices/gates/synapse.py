
import numpy as np
import matplotlib.pyplot as plt
from gates import *    

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
        
        self.Ts = []
        self.VPREX_OUT = []
        self.VPOSTX_OUT = []
        self.INC_OUT = []
        self.DEC_OUT = []
        self.IM_OUT = []
        
    def step(self, VPRE, VPOST, t):
        self.Ts.append(t)
    
        if (VPRE <= self.SPIKE_THRESH):
            self.LAST_PRE = t
        if (VPOST <= self.SPIKE_THRESH):
            self.LAST_POST = t
    
        # VPRE'
        VPREN = not_gate(self.VDD, self.VTH, VPRE)
        
        # VPREX
        '''
        DVDT = (1 / C1) * (PFET_ISD(VDD, VDD, VT, VPREX) - PFET_ISD(VDD, VPREX, VPRE, VRESET))
        DV = DVDT * dt
        VPREX = np.clip(VPREX + DV, 0, 1.0)
        VPREX_OUT.append(VPREX)
        '''
        VPREX = np.clip( self.r1 * (t - self.LAST_PRE), 0, 1.0)
        self.VPREX_OUT.append(VPREX)
        
        # VPOSTX
        '''
        DVDT = (1 / C2) * (PFET_ISD(VDD, VDD, VT, VPOSTX) - PFET_ISD(VDD, VPOSTX, VPOST, VRESET))
        DV = DVDT * dt
        VPOSTX = np.clip(VPOSTX + DV, 0, 1.0)
        VPOSTX_OUT.append(VPOSTX)
        '''
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
            
        IM = (VP - VM) / 1
        self.IM_OUT.append(IM)
    
        
# sim params
T = 2e-3
steps = 1000
dt = T / steps

# sweep over the pre synaptic input voltage.
VPRE_IN = np.concatenate(( np.linspace(1, 1, 250), np.linspace(0.32, 0.32, 20), np.linspace(1, 1, 1500) ))
VPOST_IN = np.concatenate(( np.linspace(1, 1, 350), np.linspace(0.32, 0.32, 20), np.linspace(1, 1, 1400) ))

syn = Synapse()
for step in range(steps):
    vpre = VPRE_IN[step] 
    vpost = VPOST_IN[step]
    syn.step(vpre, vpost, step * dt)

plt.subplot(2,2,1)
plt.plot(syn.Ts, syn.VPREX_OUT, syn.Ts, syn.VPOSTX_OUT)

plt.subplot(2,2,2)
plt.plot(syn.Ts, syn.INC_OUT, syn.Ts, syn.DEC_OUT)

plt.subplot(2,2,3)
plt.plot(syn.Ts, syn.IM_OUT)

# plt.subplot(2,2,4)

plt.show()







