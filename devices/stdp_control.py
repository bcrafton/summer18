
import numpy as np
import matplotlib.pyplot as plt

def PFET_IDS(VS, VG, VD):

    VGS = VG - VS
    VDS = VD - VS

    VTH = 0.6
    K = 20.0

    if (-VGS >= -VTH):
        IDS = 0
    elif (0 < -VDS) and (-VDS < -VGS + VTH):
        IDS = -K * ((VGS - VTH) * VDS - VDS ** 2 / 2)
    else:
        IDS = (-K/2) * (VGS - VTH) ** 2
        
    return IDS
    
T = 1
steps = 1000
dt = T / steps

# params from Purdue / Nature
VDD = 1.0
VRESET = 0.320
VT = 0.975
C = 500e-15

# initial prog voltage
VPROG = 1.0

# sweep over the pre synaptic input voltage.
VPRE_IN = np.linspace(0, 2.4, steps)

VPROG_OUT = []

for step in range(steps):
    t = step * dt
    
    VPRE = VPRE_IN[step]
    DVDT = (1 / C) * (PFET_IDS(VDD, VT, VPROG) - PFET_IDS(VPROG, VPRE, VRESET))
    DV = DVDT * dt
    
    VPROG += DV
    VPROG_OUT.append(VPROG)
    
plt.plot(VPRE_IN, VPROG_OUT)
plt.show()
