
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import cPickle as pickle
import gzip

#############
def load_data():
  global training_set, training_labels, testing_set, testing_labels
  f = gzip.open('mnist.pkl.gz', 'rb')
  train, valid, test = pickle.load(f)
  f.close()

  [training_set, training_labels] = train
  [validation_set, validation_labels] = valid
  [testing_set, testing_labels] = test

  for i in range( len(training_set) ):
    training_set[i] = training_set[i].reshape(28*28)

  for i in range( len(testing_set) ):
    testing_set[i] = testing_set[i].reshape(28*28)
#############
class LIF_group:
    def __init__(self, N, tau, vthr, vrest):
    
        self.N = N
        self.vthr = vthr
        self.vrest = vrest
    
        GMEAN = 100.0
        GSTD = 10.0

        self.ge_tau = 1e-3
        self.gi_tau = 2e-3
        self.tau = tau
        
        self.ge = np.zeros(shape=(N))
        self.gi = np.zeros(shape=(N))
        self.v = np.zeros(shape=(N))
        
        self.Vs = []
        
        print self.vthr
        
    def step(self, dt, Iine, Iini=0):
        gedt = -(self.ge / self.ge_tau * dt) + Iine
        self.ge = self.ge + gedt
        
        gidt = -(self.gi / self.gi_tau * dt) + Iini
        self.gi = self.gi + gidt
        
        IsynE = self.ge * -self.v
        IsynI = self.gi * (-0.085 - self.v)
        
        dvdt = ((self.vrest - self.v) + (IsynE + IsynI)) / self.tau        
        dv = dvdt * dt
        self.v += dv
        
        spkd = self.v > self.vthr
        nspkd = self.v < self.vthr
        self.v = (self.v * nspkd) + (self.v * spkd * self.vrest)
        self.ge = self.ge * nspkd
        self.gi = self.gi * nspkd
        
        # self.Vs.append(self.v[0])
        self.Vs.append(np.copy(self.v))
        
        return spkd
        
    def reset(self):
        self.ge = np.zeros(shape=(N))
        self.gi = np.zeros(shape=(N))
        self.v = np.zeros(shape=(N))
        self.Vs = []
#############
N = 1

T = 3
dt = 1e-4
steps = int(T / dt)
Ts = np.linspace(0, T, steps)

NUM_EX = 1

load_data()

w = np.load('XeAe.npy')
wei = np.load('AeAi.npy')
wie = np.load('AiAe.npy') 
theta = np.load('theta_A.npy')

lif = LIF_group(N, 1e-1, theta[0] - 20e-3 - 52e-3, -65e-3)

#############

I = np.zeros(shape=(N, 1))
Iie = np.zeros(shape=(N, 1))
Iei = np.zeros(shape=(N, 1))

spk_count = np.zeros(shape=(NUM_EX, N))

for s in range(steps):
    t = Ts[s]
    
    rates = 1
    
    '''
    spk = np.random.rand(1, 28*28) < rates * dt
    I = np.dot(spk, w)
    if (I.flatten()[0] > 0):
        print "spk"
    '''
    
    spk = np.random.rand() < rates * dt
    I = spk * 128.0
    
    if spk:
        print "spk"
    
    spkd = lif.step(dt, I, 0)
    spk_count += spkd
     
#############
print spk_count

plt.plot(Ts, lif.Vs)
plt.show()
#############


















