
# THEANO_FLAGS=device=cuda0 python th_memristor.py
# python th_memristor.py

import theano
import theano.tensor as TT
import numpy as np
import pylab as plt
import time

W =   TT.fmatrix("W")
V =   TT.fvector("V")

W0 = np.ones(shape=(100, 100)) * 5e-9
W0 = W0.astype('f')

steps = 10000
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

result, updates = theano.scan(fn=gradient, outputs_info=[W], sequences=V, n_steps=steps)
func = theano.function(inputs=[V, W], outputs=result, updates=updates)
                       
VIN = np.sin(np.linspace(0, T, steps)).astype('f')

start = time.time()
WS = func(VIN, W0)
end = time.time()
print end - start

plt.plot(VIN, WS[:, 0, 0])
plt.show()














