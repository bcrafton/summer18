
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
    def __init__(self, N, adapt, tau, theta, vthr, vrest, vreset, refrac_per, i_offset, tc_theta, theta_plus_e):
    
        self.N = N
        self.adapt = adapt
        self.theta = theta
        self.vthr = vthr
        self.vrest = vrest
        self.vreset = vreset
        self.refrac_per = refrac_per
        self.i_offset = i_offset
        self.tc_theta = tc_theta
        self.theta_plus_e = theta_plus_e

        self.ge_tau = 1e-3
        self.gi_tau = 2e-3
        self.tau = tau
        
        self.ge = np.zeros(shape=(N))
        self.gi = np.zeros(shape=(N))
        self.v = np.ones(shape=(N)) * self.vreset
        self.last_spk = np.ones(shape=(N)) * -1
        
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
        
        spkd = self.v > (self.theta + self.vthr)
        nspkd = self.v < (self.theta + self.vthr)
        
        self.last_spk = self.last_spk * nspkd
        self.last_spk += spkd * t
        
        self.v = self.v * nspkd 
        self.v += spkd * self.vreset
        
        self.ge = self.ge * nspkd
        self.gi = self.gi * nspkd
        
        if self.adapt:
            dtheta_dt = -self.theta / self.tc_theta
            self.theta = self.theta + dtheta_dt * dt
            self.theta = self.theta + spkd * self.theta_plus_e

        return spkd
        
    def reset(self):
        self.ge = np.zeros(shape=(N))
        self.gi = np.zeros(shape=(N))
        self.v = np.ones(shape=(N)) * self.vreset
        self.last_spk = np.ones(shape=(N)) * -1
        
#############
class Synapse_group:
    def __init__(self, N, M, w, stdp, tc_pre_ee, tc_post_1_ee, tc_post_2_ee, nu_ee_pre, nu_ee_post, wmax_ee):
    
        # group level variables
        self.N = N
        self.M = M
        self.stdp = stdp
        self.tc_pre_ee = tc_pre_ee
        self.tc_post_1_ee = tc_post_1_ee
        self.tc_post_2_ee = tc_post_2_ee
        self.nu_ee_pre = nu_ee_pre
        self.nu_ee_post = nu_ee_post
        self.wmax_ee = wmax_ee
        
        # synapse level variables
        self.w = w
        
        # pre level variables
        self.pre = np.zeros(self.N)
        self.last_pre = np.ones(self.N) * -1
        
        # post level variables
        self.post1 = np.zeros(self.M)
        self.post2 = np.zeros(self.M)
        self.last_post = np.ones(self.M) * -1
        
    def step(self, t, dt, pre_spk, post_spk): # we are skiping event driven parts for now
    
        I = np.dot(np.transpose(pre_spk), self.w)
    
        got_pre = np.any(pre_spk)
        got_post = np.any(post_spk)

        if (got_pre or got_post):
            dpre_dt = -self.pre / self.tc_pre_ee * (t - self.last_pre)
            dpost1_dt = -self.post1 / self.tc_post_1_ee * (t - self.last_post)
            dpost2_dt = -self.post2 / self.tc_post_2_ee * (t - self.last_post)
            
            self.pre += dpre_dt
            self.post1 += dpost1_dt
            self.post2 += dpost2_dt

            self.pre = np.clip(self.pre + pre_spk, 0, 1.0)
            self.post1 = np.clip(self.post1 + post_spk, 0, 1.0)
            self.post2 = np.clip(self.post2 + post_spk, 0, 1.0)
            
            npre_spk = pre_spk == 0
            self.last_pre = self.last_pre * npre_spk
            self.last_pre += pre_spk * t
            
            npost_spk = post_spk == 0
            self.last_post = self.last_post * npost_spk
            self.last_post += post_spk * t
            
        if (got_pre and np.any(self.post1 > 0)):
            self.w = np.clip(self.w + self.nu_ee_pre * self.post1, 0, self.wmax_ee)

        if (got_post):
            self.w = np.clip(self.w + self.nu_ee_post * np.copy(self.pre).reshape(self.N, 1) * np.copy(self.post2).reshape(1, self.M), 0, self.wmax_ee)

        return I
        
    def reset(self):
        # pre level variables
        self.pre = np.zeros(self.N)
        self.last_pre = np.ones(self.N) * -1
        
        # post level variables
        self.post1 = np.zeros(self.M)
        self.post2 = np.zeros(self.M)
        self.last_post = np.ones(self.M) * -1
        
        # normalize w
        col_sum = np.sum(self.w, axis=0)
        col_factor = 78.0 / col_sum
        for i in range(self.M):
            self.w[:, i] *= col_factor[i]
        
#############
N = 400

T = 0.35
dt = 1e-4
steps = int(T / dt)
Ts = np.linspace(0, T, steps)

NUM_EX = args.examples

load_data()

# w = np.load('./weights/XeAe.npy')
w = np.load('./random/XeAe.npy')
wei = np.load('./random/AeAi.npy')
wie = np.load('./random/AiAe.npy') 
# theta = np.load('./weights/theta_A.npy')
theta = np.ones(N) * 20e-3

Syn = Synapse_group(784, 400, w, True, 20e-3, 20e-3, 40e-3, 1e-4, 1e-2, 1.0)
lif_exc = LIF_group(N, True, 1e-1, theta, -20e-3 - 52e-3, -65e-3, -65e-3, 5e-3, -100e-3, 1e7*1e-3, 0.05e-3)
lif_inh = LIF_group(N, False, 1e-2, 0, -40e-3, -60e-3, -45e-3, 2e-3, -85e-3, 1e7*1e-3, 0.05e-3)

#############

I = np.zeros(shape=(N, 1))
Iie = np.zeros(shape=(N, 1))
Iei = np.zeros(shape=(N, 1))

spk_count = np.zeros(shape=(NUM_EX, N))
labels = np.zeros(NUM_EX)

print "starting sim"
start = time.time()

for ex in range(NUM_EX):
    
    spks = 0
    prev_spks = spks
    input_factor = 2
    
    spkd = np.zeros(N)
    
    #############
    while spks < 5:
        #############
        print ex, np.sum(spk_count), input_factor
        print np.sum(spk_count, axis=0)
        
        spk_count[ex] = 0
        for s in range(steps):
            t = Ts[s]
            
            rates = training_set[ex] * 32.0 * input_factor
            spk = np.random.rand(784) < rates * dt
            
            I = Syn.step(t, dt, spk, spkd)
            spkd = lif_exc.step(t, dt, I.flatten(), Iie.flatten())
            
            spk_count[ex] += spkd
            labels[ex] = training_labels[ex]
            
            Iei = np.dot(np.transpose(spkd), wei)
            spkd = lif_inh.step(t, dt, Iei.flatten())
            
            Iie = np.dot(np.transpose(spkd), wie)
        #############
        lif_exc.reset()
        lif_inh.reset()
        Syn.reset()
        prev = spks
        spkd = np.zeros(N)
        spks = np.sum(spk_count[ex]) - prev
        if spks < 5:
            input_factor *= 2
    #############

end = time.time()
print ("total time taken: " + str(end - start))

np.save('XeAe_trained1', Syn.w)
np.save('theta', lif_exc.theta)
#############



