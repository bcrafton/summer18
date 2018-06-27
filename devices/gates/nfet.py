
import numpy as np
import matplotlib.pyplot as plt

def NFET_IDS(VD, VG, VS):

    VDS = VD - VS
    VGS = VG - VS

    VTH = 0.423
    K = 0.4
    UT = 0.025
    I0 = 1e-10

    if (VGS <= VTH):
        if (VDS > 4 * UT):
            IDS = I0 * np.exp((K * VG - VS) / UT)
        else:
            IDS = I0 * np.exp((K * VG - VS) / UT) * (1 - np.exp(-VDS / UT))
            
    elif (VGS > VTH) and (VDS <= VGS - VTH):
        IDS = K * (VGS - VTH) * VDS - VDS ** 2
        
    elif (VGS > VTH) and (VDS > VGS - VTH):
        IDS = K * (VGS - VTH) ** 2
        
    else:
        print "should not get here"
        assert(False)
        
    return IDS
    
T = 1
steps = 1000
dt = T / steps

VDD = 1.0
VD = VDD
VG_SWEEP = np.linspace(0, 0.4, steps)
VS = 0
IDS_OUT = []

for step in range(steps):
    t = step * dt
    VG = VG_SWEEP[step]
    IDS = NFET_IDS(VD, VG, VS)
    IDS_OUT.append(IDS)

plt.plot(VG_SWEEP, IDS_OUT)
plt.show()
