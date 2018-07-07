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

# load_data()

#############

T = 0.35
dt = 1e-4
steps = int(T / dt)
Ts = np.linspace(0, T, steps)

#############

#rates = training_set[0]
#rates = rates / np.max(rates)

ge = 0
gi = 0
v = 0

Vs = []

#############

for s in range(steps):
    t = Ts[s]
    
    rate = 5
    w = 1
    ge_tau = 1e-3
    neuron_tau = 1e-1
    vrest = 0
    thresh = 0.85
    
    fired = np.random.rand() < rate * dt
    
    gedt = ge - ge / ge_tau * dt
    ge = np.min(ge, 0)
    ge += fired * w
    
    IsynE = ge 
    
    dvdt = (vrest - v + IsynE) / neuron_tau
    dv = dvdt * dt
    v += dv
    v = vrest if (v > thresh) else v
    
    Vs.append(v)
    
#############
    
plt.plot(Ts, Vs)
plt.show()
    
    
