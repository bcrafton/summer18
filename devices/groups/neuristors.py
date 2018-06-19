
import numpy as np
import pylab as plt

def mott(v, state):
    if (state == 0 and v <= 1.0):
        r = 1e6
        state = 0
    elif (state == 0 and v > 1.0):
        r = 5e4
        state = 1
    elif (state == 1 and v >= 0.5):
        r = 5e4
        state = 1
    elif (state == 1 and v < 0.5):
        r = 1e6
        state = 0
    else:
        print ("this should never happen")
        
    return r, state

class Neuristors:
    
    def __init__(self, N, M):
    
        C1 = 3e-9
        C2 = 2e-9
        R1 = 1e5
        R2 = 1e5
        VDC1 = -0.9
        VDC2 = 0.9
        RL = 1e9

        RS1 = 10e6
        RS1_STATE = 0

        RS2 = 10e6
        RS2_STATE = 0

        steps = 1500
        T = 1.5e-2
        # dt = 1e-5
        dt = T / steps

        t_in = np.linspace(0, T, steps)
        i_in = np.concatenate(( np.linspace(0, 0, 500), np.linspace(1e-6, 1e-6, 500), np.linspace(0, 0, 500) ))

        V1 = 0
        V2 = 0

        V1s = []
        V2s = []

        RS1s = []
        RS2s = []

for i in range(steps):
    dv1dt = (1 / C1) * ( i_in[i] - (1 / RS1) * (V1 - VDC1) - (1 / R2) * (V1 - V2) )
    dv2dt = (1 / C2) * ( (1 / R2) * (V1 - V2) - (1 / RS2) * (V2 - VDC2) - (1 / RL) * V2 )
    
    dv1 = dv1dt * dt
    dv2 = dv2dt * dt
    
    V1 += dv1
    V2 += dv2
    
    RS1, RS1_STATE = mott(V1 - VDC1, RS1_STATE)
    RS2, RS2_STATE = mott(VDC2 - V2, RS2_STATE)
    
    V1s.append(V1)
    V2s.append(V2)
    RS1s.append(RS1)
    RS2s.append(RS2)

'''
plt.subplot(2,2,1)
plt.plot(t_in, V2s)

plt.subplot(2,2,2)
plt.plot(t_in, i_in)

plt.show()
'''









    
