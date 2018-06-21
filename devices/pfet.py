
import numpy as np
import matplotlib.pyplot as plt

def PFET_IDS(VGS_PRE, VDS_PRE):

    VTH_PRE = 0.6
    K = 20.0

    if (-VGS_PRE >= -VTH_PRE):
        IDS_PRE = 0
    elif (0 < -VDS_PRE) and (-VDS_PRE < -VGS_PRE + VTH_PRE):
        IDS_PRE = -K * ((VGS_PRE - VTH_PRE) * VDS_PRE - VDS_PRE ** 2 / 2)
    else:
        IDS_PRE = (-K/2) * (VGS_PRE - VTH_PRE) ** 2
        
    return IDS_PRE
    
T = 1
steps = 1000
dt = T / steps

VDS = 2.4
VGS_SWEEP = np.linspace(0, 2.4, steps)

ids_out = []

for step in range(steps):
    t = step * dt
    VGS = VGS_SWEEP[step]
    ids = PFET_IDS(VGS, VDS)
    ids_out.append(ids)

plt.plot(VGS_SWEEP, ids_out)
plt.show()
