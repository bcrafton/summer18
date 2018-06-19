
import numpy as np
import pylab as plt

def mott(v, state):

    shape = np.shape(v)
    assert(shape == np.shape(state))
    r = np.zeros(shape=shape)

    off    = v <= 1.0
    off_on = v > 1.0
    on     = v >= 0.5
    on_off = v < 0.5

    off_state = state == 0
    on_state = state == 1
    
    case1 = off_state * off
    case2 = off_state * off_on
    case3 = on_state * on
    case4 = on_state * on_off

    r[np.where(case1 == 1)] = 10e6
    r[np.where(case2 == 1)] = 5e4
    r[np.where(case3 == 1)] = 5e4
    r[np.where(case4 == 1)] = 10e6

    state[np.where(case1 == 1)] = 0
    state[np.where(case2 == 1)] = 1
    state[np.where(case3 == 1)] = 1
    state[np.where(case4 == 1)] = 0
        
    return r, state

class Neuristors:
    
    def __init__(self, N, M):
    
        self.C1 = 3e-9
        self.C2 = 2e-9
        self.R1 = 1e5
        self.R2 = 1e5
        self.VDC1 = -0.9
        self.VDC2 = 0.9
        self.RL = 1e9

        self.RS1 = np.ones(shape=(N, M)) * 10e6
        self.RS1_STATE = np.zeros(shape=(N, M))
        self.RS2 = np.ones(shape=(N, M)) * 10e6
        self.RS2_STATE = np.zeros(shape=(N, M))

        self.V1 = np.zeros(shape=(N, M))
        self.V2 = np.zeros(shape=(N, M))
        
    def step(self, I, dt):
        
        dv1dt = (1 / self.C1) * ( I - (1 / self.RS1) * (self.V1 - self.VDC1) - (1 / self.R2) * (self.V1 - self.V2) )
        dv2dt = (1 / self.C2) * ( (1 / self.R2) * (self.V1 - self.V2) - (1 / self.RS2) * (self.V2 - self.VDC2) - (1 / self.RL) * self.V2 )

        dv1 = dv1dt * dt
        dv2 = dv2dt * dt
        
        self.V1 += dv1
        self.V2 += dv2
        
        self.RS1, self.RS1_STATE = mott(self.V1 - self.VDC1, self.RS1_STATE)
        self.RS2, self.RS2_STATE = mott(self.VDC2 - self.V2, self.RS2_STATE)
        
        return self.V2

##################################

steps = 1500
T = 1.5e-2
dt = T / steps

N = Neuristors(10, 10)

Ts = np.linspace(0, T, steps)
Is = np.concatenate(( np.linspace(0, 0, 500), np.linspace(1e-6, 1e-6, 500), np.linspace(0, 0, 500) ))
Vs = np.zeros(shape=(1500))

for t in range(steps):
    I = Is[t] 
    V = N.step(I, dt)
    Vs[t] = V[0, 0]
    
plt.subplot(2,2,1)
plt.plot(Ts, Vs)

plt.subplot(2,2,2)
plt.plot(Ts, Is)

plt.show()


##################################







    
