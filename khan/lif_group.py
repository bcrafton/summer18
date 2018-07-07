
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import cPickle as pickle
import gzip

#############
'''
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
'''

#############

class LIF_group:
    def __init__(self, N):
    
        GMEAN = 100.0
        GSTD = 10.0
    
        self.vrest = 0.0
        self.vthr = 0.85
        self.ge_tau = 1e-3
        self.gi_tau = 2e-3
        self.tau = 1e-1
        
        self.ge = np.zeros(N)
        self.gi = np.zeros(N)
        self.v = np.zeros(N)
        self.w = np.random.normal(GMEAN, GSTD, size=(N))
        
        self.Vs = []
        
    def step(self, spk, dt):
        nspkd = self.v < self.vthr
        self.v = self.v * nspkd + self.vrest
        
        gedt = -(self.ge / self.ge_tau * dt) + spk * self.w
        self.ge = self.ge + gedt
        
        IsynE = self.ge 
        
        dvdt = (self.vrest - self.v + IsynE) / self.tau        
        dv = dvdt * dt
        self.v += dv

        self.Vs.append(self.v)

#############
# load_data()
#############

N = 5

T = 0.35
dt = 1e-4
steps = int(T / dt)
Ts = np.linspace(0, T, steps)

lif = LIF_group(N)

#############
for s in range(steps):
    t = Ts[s]
    
    rate = 5
    
    # they all coming through same neuron.
    spk = np.random.rand() < rate * dt
    
    lif.step(spk, dt)
#############
plt.plot(Ts, lif.Vs)
plt.show()
#############



