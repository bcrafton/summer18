import theano
import theano.tensor as TT
import numpy as np
import pylab as plt

'''
DT =   TT.fscalar("DT")
U =    TT.fscalar("U")
D =    TT.fscalar("D")
RON =  TT.fscalar("RON")
ROFF = TT.fscalar("ROFF")
P =    TT.fscalar("P")
F =    TT.fscalar("F")
dwdt = TT.fscalar("dwdt")
'''

W =   TT.fscalar("W")
V =   TT.fvector("V")

# I =   TT.fscalar("Is")
# R =   TT.fscalar("Rs")

'''
U = 1e-16
D = 10e-9
W0 = 5e-9
RON = 5e4
ROFF = 1e6
P = 5
'''

W0 = 5e-9

steps = 1500
T = 2 * np.pi
dt = T / steps 

def gradient (V, W):
    U = 1e-16
    D = 10e-9
    RON = 5e4
    ROFF = 1e6
    P = 5

    R = RON * (W / D) + ROFF * (1 - (W / D))
    I = V / R
    
    F = 1 - (2 * (W / D) - 1) ** (2 * P)
    dwdt = ((U * RON * I) / D) * F
    W += dwdt * dt

    return W

result, updates = theano.scan(fn=gradient,
                              outputs_info = [W],
                              sequences = V,
                              n_steps=steps)


func = theano.function(inputs=[V, W], 
                       outputs=result, 
                       updates=updates)
                       
VIN = np.sin(np.linspace(0, T, steps)).astype('f')
WS = func(VIN, W0)

plt.plot(VIN, WS)
plt.show()















