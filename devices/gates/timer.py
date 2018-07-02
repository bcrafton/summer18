
import numpy as np
import matplotlib.pyplot as plt
from gates import *

def PFET_ISD(VDD, VS, VG, VD):

    VDS = VD - VS
    VGS = VG - VS
    
    VSD = VS - VD
    VSG = VS - VG

    VTH = -0.365
    ABS_VTH = np.absolute(VTH)
    
    K = 0.4
    UT = 0.025
    I0 = 2e-12

    if (VSG <= ABS_VTH):
        if (VSD >= 4 * UT):
            # print ("sub-sat")
            ISD = I0 * np.exp(K * (VDD - VG - VTH) / UT)
        else:
            # print ("sub")
            # Hasler has (1 - np.exp(-VDD / UT)) because saturation
            # but we have it a little differently because we not in saturation here.
            ISD = I0 * np.exp(K * (VDD - VG - VTH) / UT) # * (1 - np.exp(-VDS / UT))
        
    elif (VSG > ABS_VTH) and (VSD <= VSG - ABS_VTH):
        # print ("linear")
        ISD = K * (VSG - ABS_VTH) * VSD - VSD ** 2 / 2 
        
    elif (VSG > ABS_VTH) and (VSD > VSG - ABS_VTH):
        # print ("sat")
        ISD = 0.5 * K * (VSG - ABS_VTH) ** 2
        
    else:
        print "this should never happen"
        assert(False)
        
    return ISD
    
T = 2e-3
steps = 1000
dt = T / steps

VDD = 1.0
VRESET = 0.32
VT = 0.975
C1 = 500e-15
C2 = 250e-15

# initial prog voltage
VPREX = 1.0
VPOSTX = 1.0

# sweep over the pre synaptic input voltage.
VPRE_IN = np.concatenate(( np.linspace(1, 1, 250), np.linspace(0.32, 0.32, 20), np.linspace(1, 1, 1500) ))
VPOST_IN = np.concatenate(( np.linspace(1, 1, 350), np.linspace(0.32, 0.32, 20), np.linspace(1, 1, 1400) ))

Ts = []
VPREX_OUT = []
VPOSTX_OUT = []
INC_OUT = []
DEC_OUT = []

VTH = 0.5

for step in range(steps):
    t = step * dt
    Ts.append(t)
    
    # VPRE
    VPRE = VPRE_IN[step]
    
    # VPOST
    VPOST = VPOST_IN[step]
    
    # VPRE'
    VPREN = not_gate(VDD, VTH, VPRE)
    
    # VPREX
    DVDT = (1 / C1) * (PFET_ISD(VDD, VDD, VT, VPREX) - PFET_ISD(VDD, VPREX, VPRE, VRESET))
    DV = DVDT * dt
    VPREX = np.clip(VPREX + DV, 0, 1.0)
    VPREX_OUT.append(VPREX)
    
    # VPOSTX
    DVDT = (1 / C2) * (PFET_ISD(VDD, VDD, VT, VPOSTX) - PFET_ISD(VDD, VPOSTX, VPOST, VRESET))
    DV = DVDT * dt
    VPOSTX = np.clip(VPOSTX + DV, 0, 1.0)
    VPOSTX_OUT.append(VPOSTX)

    # VPREX nor VPOSTX
    INC = nor_gate(VDD, VTH, VPREX, VPOSTX)
    INC_OUT.append(INC)
    
    # VPREX' nor VPOSTX
    VPREXN = not_gate(VDD, VTH, VPREX)
    DEC = nor_gate(VDD, VTH, VPREXN, VPOSTX)
    DEC_OUT.append(DEC)
    
    # INC or VPRE'
    READ = or_gate(VDD, VTH, INC, VPREN)

plt.subplot(2,2,1)
plt.plot(Ts, VPREX_OUT, Ts, VPOSTX_OUT)

plt.subplot(2,2,2)
plt.plot(Ts, INC_OUT, Ts, DEC_OUT)

# plt.subplot(2,2,3)

# plt.subplot(2,2,4)

plt.show()







