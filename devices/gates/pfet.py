
import numpy as np
import matplotlib.pyplot as plt

def PFET_ISD(VDD, VS, VG, VD):

    VDS = VD - VS
    VGS = VG - VS
    
    VSD = VS - VD
    VSG = VS - VG

    VTH = -0.365
    ABS_VTH = np.absolute(VTH)
    
    K = 0.4
    UT = 0.025
    I0 = 1e-10

    if (VSG <= ABS_VTH):
        if (VSD >= 4 * UT):
            # print ("sub-sat")
            ISD = I0 * np.exp(K * (VDD - VG - VTH) / UT)
        else:
            # print ("sub")
            # Hasler has (1 - np.exp(-VDD / UT)) because saturation
            # but we have it a little differently because we not in saturation here.
            ISD = I0 * np.exp(K * (VDD - VG - VTH) / UT) * (1 - np.exp(-VDS / UT))
        
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
    
T = 1
steps = 50
dt = T / steps

VDD = 1.0
VS = VDD
VG_SWEEP = np.linspace(0.0, 1.0, steps)
VD = 0.320

IDS_OUT = []

for step in range(steps):
    t = step * dt
    VG = VG_SWEEP[step]
    IDS = PFET_ISD(VDD, VS, VG, VD)
    IDS_OUT.append(IDS)

plt.plot(VG_SWEEP, IDS_OUT)
plt.show()
