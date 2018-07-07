
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
    def __init__(self, N):
    
        GMEAN = 100.0
        GSTD = 10.0
    
        self.vrest = 0.0
        self.vthr = 0.85
        self.ge_tau = 1e-3
        self.gi_tau = 2e-3
        self.tau = 1e-1
        
        self.ge = np.zeros(shape=(N))
        self.gi = np.zeros(shape=(N))
        self.v = np.zeros(shape=(N))
        
        self.Vs = []
        
    def step(self, Iin, dt):
        nspkd = self.v < self.vthr
        self.v = self.v * nspkd + self.vrest
        
        gedt = -(self.ge / self.ge_tau * dt) + Iin
        self.ge = self.ge + gedt
        
        IsynE = self.ge 
        
        dvdt = (self.vrest - self.v + IsynE) / self.tau        
        dv = dvdt * dt
        self.v += dv

        # self.Vs.append( self.v )
        
    def reset(self):
        self.ge = np.zeros(shape=(N))
        self.gi = np.zeros(shape=(N))
        self.v = np.zeros(shape=(N))
        self.Vs = []
        
#############
# load_data()
#############
N = 400

T = 0.35
dt = 1e-4
steps = int(T / dt)
Ts = np.linspace(0, T, steps)

NUM_EX = 100

load_data()
w = np.load('XeAe.npy')

lif = LIF_group(N)
#############

for ex in range(NUM_EX):
    print ex
    #############
    for s in range(steps):
        t = Ts[s]
        
        rates = training_set[ex] * 32.0
        
        spk = np.random.rand(1, 28*28) < rates * dt
        Iin = np.dot(spk, w)
        Iin = Iin.flatten()
        
        lif.step(Iin, dt)
    #############
    lif.reset()
     
#############
# plt.plot(Ts, lif.Vs)
# plt.show()
#############



