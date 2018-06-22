
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
    I0 = 1e-7

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
    
T = 1e-6
steps = 1000
dt = T / steps

# params from Purdue / Nature
VDD = 1.0
VRESET = 0.320
VT = 0.975
C = 500e-15

# initial prog voltage
VPROG = 0.0

# sweep over the pre synaptic input voltage.
# VPRE_IN = np.linspace(0, 1.0, steps)
# VPRE_IN = np.concatenate(( np.linspace(1, 1, 250), np.linspace(0.320, 0.320, 250), np.linspace(1, 1, 500) ))
VPRE_IN = np.concatenate(( np.linspace(1, 1, 250), np.linspace(0.32, 0.32, 250), np.linspace(1, 1, 500) ))

Ts = []
VPROG_OUT = []

for step in range(steps):
    t = step * dt
    Ts.append(t)
    
    VPRE = VPRE_IN[step]
    DVDT = (1 / C) * (PFET_ISD(VDD, VDD, VT, VPROG) - PFET_ISD(VDD, VPROG, VPRE, VRESET))
    DV = DVDT * dt
    
    print (DV, PFET_ISD(VDD, VDD, VT, VPROG), PFET_ISD(VDD, VPROG, VPRE, VRESET))
    
    VPROG = np.clip(VPROG + DV, 0, 1.0)
    VPROG_OUT.append(VPROG)
    
plt.plot(Ts, VPROG_OUT)
plt.show()
