
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import cPickle as pickle
import gzip
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--examples', type=int, default=1000)
args = parser.parse_args()

random.seed(0)

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
    def __init__(self, N, tau, vthr, vrest, vreset, refrac_per, i_offset):
    
        self.N = N
        self.vthr = vthr
        self.vrest = vrest
        self.vreset = vreset
        self.refrac_per = refrac_per
        self.i_offset = i_offset
    
        GMEAN = 100.0
        GSTD = 10.0

        self.ge_tau = 1e-3
        self.gi_tau = 2e-3
        self.tau = tau
        
        self.ge = np.zeros(shape=(N))
        self.gi = np.zeros(shape=(N))
        self.v = np.ones(shape=(N)) * self.vreset
        self.last_spk = np.ones(shape=(N)) * -1
        
        self.Vs = []
        
    def step(self, t, dt, Iine, Iini=0):
        gedt = -(self.ge / self.ge_tau * dt) + Iine
        self.ge = self.ge + gedt
        
        gidt = -(self.gi / self.gi_tau * dt) + Iini
        self.gi = self.gi + gidt
        
        IsynE = self.ge * -self.v
        IsynI = self.gi * (self.i_offset - self.v)
        
        nrefrac = (t - self.last_spk - self.refrac_per) > 0
        
        dvdt = ((self.vrest - self.v) + (IsynE + IsynI)) / self.tau        
        dv = dvdt * dt
        self.v += nrefrac * dv
        
        spkd = self.v > self.vthr
        nspkd = self.v < self.vthr
        
        self.last_spk = self.last_spk * nspkd
        self.last_spk += spkd * t
        
        self.v = self.v * nspkd 
        self.v += spkd * self.vreset
        
        self.ge = self.ge * nspkd
        self.gi = self.gi * nspkd
        
        # self.Vs.append( np.copy(self.v) )
        
        return spkd
        
    def reset(self):
        self.ge = np.zeros(shape=(N))
        self.gi = np.zeros(shape=(N))
        self.v = np.ones(shape=(N)) * self.vreset
        self.last_spk = np.ones(shape=(N)) * -1
        # self.Vs = []
        
#############
N = 400

T = 0.35
dt = 1e-4
steps = int(T / dt)
Ts = np.linspace(0, T, steps)

NUM_EX = args.examples

load_data()

w = np.load('./weights/XeAe.npy')
wei = np.load('./weights/AeAi.npy')
wie = np.load('./weights/AiAe.npy') 
theta = np.load('./weights/theta_A.npy')

lif_exc = LIF_group(N, 1e-1, theta - 20e-3 - 52e-3, -65e-3, -65e-3, 5e-3, -100e-3)
lif_inh = LIF_group(N, 1e-2, -40e-3, -60e-3, -45e-3, 2e-3, -85e-3)

#############

I = np.zeros(shape=(N, 1))
Iie = np.zeros(shape=(N, 1))
Iei = np.zeros(shape=(N, 1))

spk_count = np.zeros(shape=(NUM_EX, N))
labels = np.zeros(NUM_EX)

print "starting sim"
start = time.time()

runtimes = open("runtimes_train", "w")
runtimes.write("runtimes training\n")
runtimes.close()

for ex in range(NUM_EX):
    
    if (ex % 10 == 0):
        print "Example # " + str(ex) + " / " + str(NUM_EX)
        runtimes = open("runtimes_train", "a")
        runtimes.write("Example # " + str(ex) + " / " + str(NUM_EX) + " | " + str(time.time() - start) + "\n")
        runtimes.close()
    
    if (ex % 100 == 0):
        np.save('./results/train_spks_'   + str(ex), spk_count[0:ex])
        np.save('./results/train_labels_' + str(ex), labels[0:ex])
    
    spks = 0
    prev_spks = spks
    input_factor = 2
    
    #############
    while spks < 5:
        #############
        # print ex, np.sum(spk_count), input_factor
        spk_count[ex] = 0
        for s in range(steps):
            t = Ts[s]
            
            rates = training_set[ex] * 32.0 * input_factor
            spk = np.random.rand(1, 28*28) < rates * dt
            
            I = np.dot(spk, w)
            spkd = lif_exc.step(t, dt, I.flatten(), Iie.flatten())
            
            spk_count[ex] += spkd
            labels[ex] = training_labels[ex]
            
            Iei = np.dot(np.transpose(spkd), wei)
            spkd = lif_inh.step(t, dt, Iei.flatten())
            
            Iie = np.dot(np.transpose(spkd), wie)
        #############
        lif_exc.reset()
        lif_inh.reset()
        prev = spks
        spks = np.sum(spk_count[ex]) - prev
        if spks < 5:
            input_factor *= 2
    #############

end = time.time()
print ("total time taken: " + str(end - start))
runtimes = open("runtimes_train", "a")
runtimes.write("total time taken: " + str(end - start) + "\n") 
runtimes.close()
#############
# print np.sum(spk_count)
# print np.shape(spk_count)
# print np.shape(labels)

np.save('./results/train_spks_'   + str(NUM_EX), spk_count)
np.save('./results/train_labels_' + str(NUM_EX), labels)

'''
print np.sum(spk_count, axis=0)
print np.argmax(np.sum(spk_count, axis=0))
idx = np.argmax(np.sum(spk_count, axis=0))
print lif_exc.vthr[idx]

Ts = np.linspace(0, 10*T, 10*steps)
plt.plot(Ts, np.transpose(lif_exc.Vs)[idx], Ts, [lif_exc.vthr[idx]] * 10 * steps)
plt.show()
'''
#############



