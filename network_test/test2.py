
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import cPickle as pickle
import gzip
import time
import argparse
from collections import deque

np.random.seed(0)

class LIF_group:
    def __init__(self, N, adapt, tau, theta, vthr, vrest, vreset, refrac_per, tc_theta, theta_plus_e):
    
        self.N = N
        self.adapt = adapt
        self.theta = theta
        self.vthr = vthr
        self.vrest = vrest
        self.vreset = vreset
        self.refrac_per = refrac_per
        self.tc_theta = tc_theta
        self.theta_plus_e = theta_plus_e
        self.tau = tau
        
        self.v = np.ones(shape=(self.N)) * self.vreset
        self.last_spk = np.ones(shape=(self.N)) * -1
        
        self.Vs = []
        
    def step(self, t, dt, Iine, Iini=0):        
        nrefrac = (t - self.last_spk - self.refrac_per) > 0
        
        # compute derivatives
        dvdt = ((self.vrest - self.v) + (Iine - Iini)) / self.tau
        dv = dvdt * dt
        
        # update state variables
        self.v = np.clip(self.v + dv * nrefrac, 0.5 * self.vreset, 1.5 * self.vthr)
        
        self.Vs.append(self.v)
                
        # reset.
        spkd = self.v > (self.theta + self.vthr)
        nspkd = self.v < (self.theta + self.vthr)
        
        self.last_spk = self.last_spk * nspkd
        self.last_spk += spkd * t
        
        self.v = self.v * nspkd 
        self.v += spkd * self.vreset
        
        if self.adapt:
            # think its fine to do theta by itself down here...
            dtheta_dt = -self.theta / self.tc_theta
            self.theta = self.theta + dtheta_dt * dt
            self.theta = self.theta + spkd * self.theta_plus_e

        return spkd
        
    def reset(self):
        self.v = np.ones(shape=(self.N)) * self.vreset
        self.last_spk = np.ones(shape=(self.N)) * -1
        
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

        # delay.
        self.pre_spks = [np.zeros(shape=(self.N))] * int(5e-3 / 1e-4)
        
        # pre level variables
        self.last_pre = np.ones(self.N) * -1
        
        # post level variables
        self.last_post = np.ones(self.M) * -1
        
    def step(self, t, dt, pre_spk, post_spk):
    
        self.pre_spks.insert(0, pre_spk)
        I = np.dot(np.transpose(self.pre_spks.pop()), self.w)
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
        
#############
N = 24
M = 48
O = 4

dt = 1e-4
t = 0.0
steps = 0

active_T = 0.35
active_steps = int(active_T / dt)

rest_T = 0.15
rest_steps = int(rest_T / dt)

NUM_EX = 10

#############

# care about these until you get spikes.
w = np.absolute(np.random.normal(10.0, 5.0, size=(N, M)))

# if you are getting no spikes, then these dont even play a role.
wei = np.ones(shape=(M, M)) * 5.0
wie = np.ones(shape=(M, M)) * 5.0

Syn = Synapse_group(N=N,                     \
                    M=M,                     \
                    w=w,                     \
                    stdp=False,              \
                    tc_pre_ee=20e-3,         \
                    tc_post_1_ee=20e-3,      \
                    tc_post_2_ee=40e-3,      \
                    nu_ee_pre=1e-4,          \
                    nu_ee_post=1e-2,         \
                    wmax_ee=1.0)

lif_exc = LIF_group(N=M,                     \
                    adapt=False,             \
                    tau=1e-1,                \
                    theta=0.0,               \
                    vthr=5e-2,               \
                    vrest=1e-2,              \
                    vreset=1e-2,             \
                    refrac_per=5e-3,         \
                    tc_theta=1e7*1e-3,       \
                    theta_plus_e=0.05e-3)

lif_inh = LIF_group(N=M,                      \
                    adapt=False,              \
                    tau=1e-2,                 \
                    theta=0.0,                \
                    vthr=5e-2,                \
                    vrest=1e-2,               \
                    vreset=1e-2,              \
                    refrac_per=2e-3,          \
                    tc_theta=1e7*1e-3,        \
                    theta_plus_e=0.05e-3)

#############

# care about these until you get spikes.
w2 = np.absolute(np.random.normal(10.0, 5.0, size=(M, O)))

# if you are getting no spikes, then these dont even play a role.
wei2 = np.ones(shape=(O, O)) * 5.0
wie2 = np.ones(shape=(O, O)) * 5.0

Syn2 = Synapse_group(N=M,                    \
                    M=O,                     \
                    w=w2,                    \
                    stdp=False,              \
                    tc_pre_ee=20e-3,         \
                    tc_post_1_ee=20e-3,      \
                    tc_post_2_ee=40e-3,      \
                    nu_ee_pre=1e-4,          \
                    nu_ee_post=1e-2,         \
                    wmax_ee=1.0)

lif_exc2 = LIF_group(N=O,                    \
                    adapt=False,             \
                    tau=1e-1,                \
                    theta=0.0,               \
                    vthr=5e-2,               \
                    vrest=1e-2,              \
                    vreset=1e-2,             \
                    refrac_per=5e-3,         \
                    tc_theta=1e7*1e-3,       \
                    theta_plus_e=0.05e-3)

lif_inh2 = LIF_group(N=O,                     \
                    adapt=False,              \
                    tau=1e-2,                 \
                    theta=0.0,                \
                    vthr=5e-2,                \
                    vrest=1e-2,               \
                    vreset=1e-2,              \
                    refrac_per=2e-3,          \
                    tc_theta=1e7*1e-3,        \
                    theta_plus_e=0.05e-3)

#############

print "starting sim"
start = time.time()

input_spk_count = np.zeros(N)
spk_count = np.zeros(shape=(NUM_EX, M))
spk_count2 = np.zeros(shape=(NUM_EX, O))
labels = np.zeros(NUM_EX)

ex = 0
input_intensity = 2.00

while ex < NUM_EX:
    lif_exc_spkd = np.zeros(shape=(M))
    lif_inh_spkd = np.zeros(shape=(M))
    lif_exc_spkd2 = np.zeros(shape=(O))
    lif_inh_spkd2 = np.zeros(shape=(O))
    #############
    spkd = np.zeros(M)    
    for s in range(active_steps):
        t += dt
        steps += 1
        
        rates = np.random.rand(N) * input_intensity
        spk = np.random.rand(N) < rates * dt
        input_spk_count += spk
        
        I = Syn.step(t, dt, spk, lif_exc_spkd)
        Iie = np.dot(np.transpose(lif_inh_spkd), wie)
        Iei = np.dot(np.transpose(lif_exc_spkd), wei)
        
        I2 = Syn2.step(t, dt, lif_exc_spkd, lif_exc_spkd2)
        Iie2 = np.dot(np.transpose(lif_inh_spkd2), wie2)
        Iei2 = np.dot(np.transpose(lif_exc_spkd2), wei2)
        
        lif_exc_spkd = lif_exc.step(t, dt, I.flatten(), Iie.flatten())        
        lif_inh_spkd = lif_inh.step(t, dt, Iei.flatten())
        lif_exc_spkd2 = lif_exc2.step(t, dt, I2.flatten(), Iie2.flatten())        
        lif_inh_spkd2 = lif_inh2.step(t, dt, Iei2.flatten())
        
        spk_count[ex] += lif_exc_spkd
        spk_count2[ex] += lif_exc_spkd2
    #############
    for s in range(rest_steps):
        t += dt
        steps += 1
        
        spk = np.zeros(N)
        
        I = Syn.step(t, dt, spk, lif_exc_spkd)
        Iie = np.dot(np.transpose(lif_inh_spkd), wie)
        Iei = np.dot(np.transpose(lif_exc_spkd), wei)
        
        I2 = Syn2.step(t, dt, lif_exc_spkd, lif_exc_spkd2)
        Iie2 = np.dot(np.transpose(lif_inh_spkd2), wie2)
        Iei2 = np.dot(np.transpose(lif_exc_spkd2), wei2)
        
        lif_exc_spkd = lif_exc.step(t, dt, I.flatten(), Iie.flatten())        
        lif_inh_spkd = lif_inh.step(t, dt, Iei.flatten())
        lif_exc_spkd2 = lif_exc2.step(t, dt, I2.flatten(), Iie2.flatten())        
        lif_inh_spkd2 = lif_inh2.step(t, dt, Iei2.flatten())
        
        spk_count[ex] += lif_exc_spkd
        spk_count2[ex] += lif_exc_spkd2
    #############
    
    lif_exc.reset()
    lif_inh.reset()
    Syn.reset()
    lif_exc2.reset()
    lif_inh2.reset()
    Syn2.reset()
    
    print "----------"
    print ex, dt, input_intensity
    print np.sum(input_spk_count), np.sum(spk_count), np.sum(spk_count2)
    print spk_count
    
    
    '''
    if np.sum(spk_count[ex]) < 5:
        spk_count[ex] = 0
        input_intensity += 0.5
    else:
        input_intensity = 2.00
        dt = 1e-4
        ex += 1
    '''
    ex += 1

end = time.time()
print ("total time taken: " + str(end - start))

plt.plot(np.linspace(0, t, steps), lif_exc2.Vs)
plt.show()


