

import numpy as np
import pylab as plt

class Memristors:
    
    def __init__(self, N, M):
        U = 1e-16
        D = 10e-9
        W0 = 5e-9
        RON = 5e4
        ROFF = 1e6
        P = 5
        
        steps = 1500
        T = 2 * np.pi
        dt = T / steps
        t = 0
    
        W = W0
        Is = np.zeros(shape=(N, M, steps))
        Vs = np.zeros(shape=(N, M, steps))
        Rs = np.zeros(shape=(N, M, steps))
        
    # this should take a voltage matrix (N, M) input
    def step(self):
        # add some randomness here.
        V = 1 * np.sin(t * dt)

        R = RON * (W / D) + ROFF * (1 - (W / D))
        I = V / R

        F = 1 - (2 * (W / D) - 1) ** (2 * P)
        dwdt = ((U * RON * I) / D) * F
        W += dwdt * dt
        
        
