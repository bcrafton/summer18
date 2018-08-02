
import numpy as np
import random
import math
import gzip
import time
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--examples', type=int, default=10000)
parser.add_argument('--train', type=int, default=False)
args = parser.parse_args()

np.random.seed(0)

#############
def load_data():
  global training_set, training_labels, testing_set, testing_labels
  f = gzip.open('./mnist.pkl.gz', 'rb')
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
        nrefrac = (t - self.last_spk - self.refrac_per) > 0
            
        IsynE = self.ge * -self.v
        IsynI = self.gi * (self.i_offset - self.v)
        
        # compute derivatives
        dvdt = ((self.vrest - self.v) + (IsynE + IsynI)) / self.tau        
        dv = dvdt * dt
        dge = -(self.ge / self.ge_tau * dt) 
        dgi = -(self.gi / self.gi_tau * dt) 
        
        # update state variables
        self.v += nrefrac * dv
        self.ge += (dge + Iine) * nrefrac
        self.gi += (dgi + Iini) * nrefrac
                
        # reset.
        spkd = self.v > (self.theta + self.vthr)
        nspkd = self.v < (self.theta + self.vthr)
        
        self.last_spk = self.last_spk * nspkd
        self.last_spk += spkd * t
        
        self.v = self.v * nspkd 
        self.v += spkd * self.vreset
        
        self.ge = self.ge * nspkd
        self.gi = self.gi * nspkd
        
        if self.adapt:
            # think its fine to do theta by itself down here...
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
        self.last_pre = np.ones(self.N) * -1
        
        # post level variables
        self.last_post = np.ones(self.M) * -1
        
    def step(self, t, dt, pre_spk, post_spk):
    
        I = np.dot(np.transpose(pre_spk), self.w)
        dw = np.zeros(shape=(self.N, self.M))

        if self.stdp:
            got_pre = np.any(pre_spk)
            got_post = np.any(post_spk)
        
            if (got_pre):
                npre_spk = pre_spk == 0
                self.last_pre = self.last_pre * npre_spk
                self.last_pre += pre_spk * t
            
                post1 = np.exp(-(t - self.last_post) / self.tc_post_1_ee)
                # self.w = np.clip(self.w - self.nu_ee_pre * np.dot(pre_spk.reshape(self.N, 1), post1.reshape(1, self.M)), 0, self.wmax_ee)
                dw += -self.nu_ee_pre * np.dot(pre_spk.reshape(self.N, 1), post1.reshape(1, self.M))
                # self.w = np.clip(self.w + dw, 0, self.wmax_ee)

            if (got_post):
                pre = np.exp(-(t - self.last_pre) / self.tc_pre_ee)
                post2 = np.exp(-(t - self.last_post) / self.tc_post_2_ee)
                # self.w = np.clip(self.w + self.nu_ee_post * np.dot(pre.reshape(self.N, 1), post2.reshape(1, self.M) * post_spk.reshape(1, self.M)), 0, self.wmax_ee)
                dw += self.nu_ee_post * np.dot(pre.reshape(self.N, 1), post2.reshape(1, self.M) * post_spk.reshape(1, self.M))
                # self.w = np.clip(self.w + dw, 0, self.wmax_ee)

                npost_spk = post_spk == 0
                self.last_post = self.last_post * npost_spk
                self.last_post += post_spk * t
                
            self.w = np.clip(self.w + dw, 0, self.wmax_ee)

        return I
        
    def reset(self):
        # pre level variables
        self.last_pre = np.ones(self.N) * -1
        
        # post level variables
        self.last_post = np.ones(self.M) * -1 
        
        if self.stdp:
            # normalize w
            col_sum = np.sum(np.copy(self.w), axis=0)
            col_factor = 78.0 / col_sum
            for i in range(self.M):
                self.w[:, i] *= col_factor[i]
        
#############
N = 400

# default timestep in brian is 0.1ms
# original simulation uses 0.5ms
# dt = 1e-4
dt = 0.5e-3

active_T = 0.35
active_steps = int(active_T / dt)
active_Ts = np.linspace(0, active_T, active_steps)

rest_T = 0.15
rest_steps = int(rest_T / dt)
rest_Ts = np.linspace(active_T, active_T + rest_T, rest_steps)

NUM_EX = args.examples

load_data()

if args.train:
    assert(False)
else:
    w = np.load('./3k/XeAe_trained_3000.npy')
    theta = np.load('./3k/theta_trained_3000.npy')
    
wei = np.load('./random/AeAi.npy')
wie = np.load('./random/AiAe.npy')

Syn = Synapse_group(N=784,                   \
                    M=400,                   \
                    w=w,                     \
                    stdp=args.train,         \
                    tc_pre_ee=20e-3,         \
                    tc_post_1_ee=20e-3,      \
                    tc_post_2_ee=40e-3,      \
                    nu_ee_pre=1e-4,          \
                    nu_ee_post=1e-2,         \
                    wmax_ee=1.0)

lif_exc = LIF_group(N=N,                     \
                    adapt=args.train,        \
                    tau=1e-1,                \
                    theta=theta,             \
                    vthr=-20e-3 - 52e-3,     \
                    vrest=-65e-3,            \
                    vreset=-65e-3,           \
                    refrac_per=5e-3,         \
                    i_offset=-100e-3,        \
                    tc_theta=1e7*1e-3,       \
                    theta_plus_e=0.05e-3)

lif_inh = LIF_group(N=N,                      \
                    adapt=False,              \
                    tau=1e-2,                 \
                    theta=0,                  \
                    vthr=-40e-3,              \
                    vrest=-60e-3,             \
                    vreset=-45e-3,            \
                    refrac_per=2e-3,          \
                    i_offset=-85e-3,          \
                    tc_theta=1e7*1e-3,        \
                    theta_plus_e=0.05e-3)

#############

print ("starting sim")
start = time.time()
    
I = np.zeros(shape=(N, 1))
Iie = np.zeros(shape=(N, 1))
Iei = np.zeros(shape=(N, 1))

spk_count = np.zeros(shape=(NUM_EX, N))
labels = np.zeros(NUM_EX)

ex = 0
input_intensity = 2

while ex < NUM_EX:
    #############
    spkd = np.zeros(N)    
    for s in range(active_steps):
        t = active_Ts[s]
        
        if args.train:
            rates = training_set[ex] * 32.0 * input_intensity
            labels[ex] = training_labels[ex]
        else:
            rates = testing_set[ex] * 32.0 * input_intensity
            labels[ex] = testing_labels[ex]

        spk = np.random.rand(784) < rates * dt
        
        I = Syn.step(t, dt, spk, spkd)
        spkd = lif_exc.step(t, dt, I.flatten(), Iie.flatten())
        
        spk_count[ex] += spkd
        
        Iei = np.dot(np.transpose(spkd), wei)
        spkd = lif_inh.step(t, dt, Iei.flatten())
        
        Iie = np.dot(np.transpose(spkd), wie)
    #############
    for s in range(rest_steps):
        t = rest_Ts[s]
        
        spk = np.zeros(784)
        
        I = Syn.step(t, dt, spk, spkd)
        spkd = lif_exc.step(t, dt, I.flatten(), Iie.flatten())
        
        spk_count[ex] += spkd
        
        Iei = np.dot(np.transpose(spkd), wei)
        spkd = lif_inh.step(t, dt, Iei.flatten())
        
        Iie = np.dot(np.transpose(spkd), wie)
    #############
    
    lif_exc.reset()
    lif_inh.reset()
    Syn.reset()
    
    print ("----------")
    print (ex, dt, input_intensity)
    print (np.sum(spk_count))
    print (np.std(Syn.w), np.max(Syn.w), np.min(Syn.w))
    print (np.sum(spk_count, axis=0))
    
    if (ex % 1000 == 0 and args.train):
        np.save('XeAe_trained_' + str(ex), Syn.w)
        np.save('theta_trained_' + str(ex), lif_exc.theta)
    
    if np.sum(spk_count[ex]) < 5:
        spk_count[ex] = 0
        input_intensity += 0.1
    elif np.sum(spk_count[ex]) > 100 and dt > 1e-6:
        dt *= 0.5
    else:
        input_intensity = 2
        dt = 0.5e-3
        ex += 1    

end = time.time()
print ("total time taken: " + str(end - start))

if args.train:
    assert(False)
    np.save('XeAe_trained', Syn.w)
    np.save('theta_trained', lif_exc.theta)
else:
    num_assign = int(NUM_EX / 2) + int(NUM_EX % 2)
    num_test = int(NUM_EX / 2)
    
    np.save('./results/assign_spks_3000',   spk_count[0:num_assign])
    np.save('./results/assign_labels_3000', labels[0:num_assign])
    np.save('./results/test_spks_3000',     spk_count[num_assign:num_assign+num_test])
    np.save('./results/test_labels_3000',   labels[num_assign:num_assign+num_test])
#############




