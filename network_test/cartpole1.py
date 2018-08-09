
import gym
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import deque

#############

parser = argparse.ArgumentParser()
parser.add_argument('--examples', type=int, default=1000)
parser.add_argument('--train', type=int, default=False)
args = parser.parse_args()
np.random.seed(0)

#############

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
        
        # do we use eligibility trace or just pre and post ? 
        self.e = np.zeros(shape=(N, M))
        
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
                dw += -self.nu_ee_pre * np.dot(pre_spk.reshape(self.N, 1), post1.reshape(1, self.M))

            if (got_post):
                pre = np.exp(-(t - self.last_pre) / self.tc_pre_ee)
                post2 = np.exp(-(t - self.last_post) / self.tc_post_2_ee)
                dw += self.nu_ee_post * np.dot(pre.reshape(self.N, 1), post2.reshape(1, self.M) * post_spk.reshape(1, self.M))

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
        
        # zero out the e-trace
        self.e = np.zeros(shape=(self.N, self.M))
        
        # we dont need to normalize with back prop now.
        '''
        if self.stdp:
            # normalize w
            col_sum = np.sum(np.copy(self.w), axis=0)
            col_factor = 78.0 / col_sum
            for i in range(self.M):
                self.w[:, i] *= col_factor[i]
        '''

#############

def preprocess(state):
    new_state = np.zeros(16)
    
    new_state[0] = state[0]      if state[0] > 0 else 0
    new_state[1] = abs(state[0]) if state[0] < 0 else 0
    new_state[2] = np.clip(state[0] + 1, 0, 2*1)
    new_state[3] = np.clip(state[0] - 1, 0, 2*1)

    new_state[4] = state[1]      if state[1] > 0 else 0
    new_state[5] = abs(state[1]) if state[1] < 0 else 0
    new_state[6] = np.clip(state[1] + 1, 0, 2*1)
    new_state[7] = np.clip(state[1] - 1, 0, 2*1)
    
    new_state[8] = state[2]      if state[2] > 0 else 0
    new_state[9] = abs(state[2]) if state[2] < 0 else 0
    new_state[10] = np.clip(state[2] + 0.27, 0, 2*0.27)
    new_state[11] = np.clip(state[2] - 0.27, 0, 2*0.27)
    
    new_state[12] = state[3]      if state[3] > 0 else 0
    new_state[13] = abs(state[3]) if state[3] < 0 else 0
    new_state[14] = np.clip(state[3] + 1, 0, 2*1)
    new_state[15] = np.clip(state[3] - 1, 0, 2*1)

    return new_state

class Solver():
    def __init__(self, epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.999, n_episodes=1000, n_win_ticks=195, max_env_steps=200):
        self.env = gym.make('CartPole-v0')
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.max_env_steps = max_env_steps

        # layer sizes
        self.LAYER0 = 16
        self.LAYER1 = 24
        self.LAYER2 = 1
        
        # run time specs
        self.dt = 1e-4

        self.active_T = 0.35
        self.active_steps = int(self.active_T / self.dt)

        self.rest_T = 0.15
        self.rest_steps = int(self.rest_T / self.dt)
        
        # run time metrics
        self.time_elapsed = 0.0
        self.total_steps = 0.0
        
        # initial weights
        self.l0_l1_w = np.absolute(np.random.normal(10.0, 5.0, size=(self.LAYER0, self.LAYER1)))
        self.l1_exc_w = np.random.normal(5.0, 0.0, size=(self.LAYER1, self.LAYER1))
        self.l1_inh_w = np.random.normal(5.0, 0.00, size=(self.LAYER1, self.LAYER1))

        self.l1_l2_w =  np.absolute(np.random.normal(10.0, 5.0, size=(self.LAYER1, self.LAYER2)))
        self.l2_exc_w = np.random.normal(5.0, 5.0, size=(self.LAYER2, self.LAYER2))
        self.l2_inh_w = np.random.normal(5.0, 5.0, size=(self.LAYER2, self.LAYER2))

        self.l1_theta = np.ones(self.LAYER1) * 20e-3
        self.l2_theta = np.ones(self.LAYER2) * 20e-3
        
        ##########################################################

        self.l0_l1_syn = Synapse_group(N=self.LAYER0, \
                            M=self.LAYER1,            \
                            w=self.l0_l1_w,           \
                            stdp=args.train,          \
                            tc_pre_ee=20e-3,          \
                            tc_post_1_ee=20e-3,       \
                            tc_post_2_ee=40e-3,       \
                            nu_ee_pre=1e-4,           \
                            nu_ee_post=1e-2,          \
                            wmax_ee=1.0)

        self.l1_exc_lif = LIF_group(N=self.LAYER1,   \
                            adapt=args.train,        \
                            tau=1e-1,                \
                            theta=self.l1_theta,     \
                            vthr=5e-2,               \
                            vrest=1e-2,              \
                            vreset=1e-2,             \
                            refrac_per=5e-3,         \
                            tc_theta=1e7*1e-3,       \
                            theta_plus_e=0.05e-3)

        self.l1_inh_lif = LIF_group(N=self.LAYER1,    \
                            adapt=False,              \
                            tau=1e-2,                 \
                            theta=0.0,                \
                            vthr=5e-2,                \
                            vrest=1e-2,               \
                            vreset=1e-2,              \
                            refrac_per=2e-3,          \
                            tc_theta=1e7*1e-3,        \
                            theta_plus_e=0.05e-3)
                            
        ##########################################################
                            
        self.l1_l2_syn = Synapse_group(N=self.LAYER1, \
                            M=self.LAYER2,            \
                            w=self.l1_l2_w,           \
                            stdp=args.train,          \
                            tc_pre_ee=20e-3,          \
                            tc_post_1_ee=20e-3,       \
                            tc_post_2_ee=40e-3,       \
                            nu_ee_pre=1e-4,           \
                            nu_ee_post=1e-2,          \
                            wmax_ee=1.0)

        self.l2_exc_lif = LIF_group(N=self.LAYER2,   \
                            adapt=args.train,        \
                            tau=1e-1,                \
                            theta=self.l2_theta,     \
                            vthr=5e-2,               \
                            vrest=1e-2,              \
                            vreset=1e-2,             \
                            refrac_per=5e-3,         \
                            tc_theta=1e7*1e-3,       \
                            theta_plus_e=0.05e-3)

        self.l2_inh_lif = LIF_group(N=self.LAYER2,    \
                            adapt=False,              \
                            tau=1e-2,                 \
                            theta=0.0,                \
                            vthr=5e-2,                \
                            vrest=1e-2,               \
                            vreset=1e-2,              \
                            refrac_per=2e-3,          \
                            tc_theta=1e7*1e-3,        \
                            theta_plus_e=0.05e-3)

        ##########################################################
        
    def choose_action(self, spks):
        return self.env.action_space.sample() if (np.random.random() <= self.epsilon) else np.argmax(spks)
        
    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        
    def run_snn(self, state):
        # spks and currents
        l0_l1_spk = np.zeros(shape=(self.LAYER0))
        l0_l1_I = np.zeros(shape=(self.LAYER1))
        
        l1_exc_spk = np.zeros(shape=(self.LAYER1))
        l1_exc_I = np.zeros(shape=(self.LAYER1))
        l1_inh_spk = np.zeros(shape=(self.LAYER1))
        l1_inh_I = np.zeros(shape=(self.LAYER1))
        
        l1_l2_spk = np.zeros(shape=(self.LAYER1))
        l1_l2_I = np.zeros(shape=(self.LAYER2))
        
        l2_exc_spk = np.zeros(shape=(self.LAYER2))
        l2_exc_I = np.zeros(shape=(self.LAYER2))
        l2_inh_spk = np.zeros(shape=(self.LAYER2))
        l2_inh_I = np.zeros(shape=(self.LAYER2))
    
        # spk counters
        l0_spks = np.zeros(shape=(self.LAYER0))
        l1_exc_spks = np.zeros(shape=(self.LAYER1))
        l2_exc_spks = np.zeros(shape=(self.LAYER2))
        
        # intensity, could probably make this smarter
        input_intensity = 64.0
        #############
        for s in range(self.active_steps):
            t = self.time_elapsed
            self.time_elapsed += self.dt
            self.total_steps += 1
            
            rates = state
            rates *= 1.0 / np.average(rates) * input_intensity

            # l0 -> l1 connections
            # rates and np.random.rand() have to have save dimensions or u get a NxN ... not Nx1
            l0_l1_spk = np.random.rand(self.LAYER0) < rates * self.dt
            l0_l1_I = self.l0_l1_syn.step(t, self.dt, l0_l1_spk, l1_exc_spk)
            
            # l1 recurrent connections
            l1_exc_spk = self.l1_exc_lif.step(t, self.dt, l0_l1_I.flatten(), l1_inh_I.flatten())
            l1_exc_I = np.dot(np.transpose(l1_exc_spk), self.l1_exc_w)
            l1_inh_spk = self.l1_inh_lif.step(t, self.dt, l1_exc_I.flatten())
            l1_inh_I = np.dot(np.transpose(l1_inh_spk), self.l1_inh_w)
            
            # l1 -> l2 connections
            l1_l2_spk = l1_exc_spk
            l1_l2_I = self.l1_l2_syn.step(t, self.dt, l1_l2_spk, l2_exc_spk)
            
            # l2 recurrent connections
            l2_exc_spk = self.l2_exc_lif.step(t, self.dt, l1_l2_I.flatten(), l2_inh_I.flatten())
            l2_exc_I = np.dot(np.transpose(l2_exc_spk), self.l2_exc_w)
            l2_inh_spk = self.l2_inh_lif.step(t, self.dt, l2_exc_I.flatten())
            l2_inh_I = np.dot(np.transpose(l2_inh_spk), self.l2_inh_w)
            
            # spike counter updates
            l1_exc_spks += l1_exc_spk
            l2_exc_spks += l2_exc_spk
            l0_spks += l0_l1_spk
        #############
        return l1_exc_spks, l2_exc_spks
        #############

    def train_snn(self, prev_state, prev_value, state, value, reward):
        # spks and currents
        l0_l1_spk = np.zeros(shape=(self.LAYER0))
        l0_l1_I = np.zeros(shape=(self.LAYER1))
        
        l1_exc_spk = np.zeros(shape=(self.LAYER1))
        l1_exc_I = np.zeros(shape=(self.LAYER1))
        l1_inh_spk = np.zeros(shape=(self.LAYER1))
        l1_inh_I = np.zeros(shape=(self.LAYER1))
        
        l1_l2_spk = np.zeros(shape=(self.LAYER1))
        l1_l2_I = np.zeros(shape=(self.LAYER2))
        
        l2_exc_spk = np.zeros(shape=(self.LAYER2))
        l2_exc_I = np.zeros(shape=(self.LAYER2))
        l2_inh_spk = np.zeros(shape=(self.LAYER2))
        l2_inh_I = np.zeros(shape=(self.LAYER2))
    
        # spk counters
        l0_spks = np.zeros(shape=(self.LAYER0))
        l1_exc_spks = np.zeros(shape=(self.LAYER1))
        l2_exc_spks = np.zeros(shape=(self.LAYER2))
        #############
        for s in range(self.rest_steps):
            t = self.time_elapsed
            self.time_elapsed += self.dt
            self.total_steps += 1

            # l0 -> l1 connections
            # rates and np.random.rand() have to have save dimensions or u get a NxN ... not Nx1
            l0_l1_spk = np.zeros(self.LAYER0)
            l0_l1_I = self.l0_l1_syn.step(t, self.dt, l0_l1_spk, l1_exc_spk)
            
            # l1 recurrent connections
            l1_exc_spk = self.l1_exc_lif.step(t, self.dt, l0_l1_I.flatten(), l1_inh_I.flatten())
            l1_exc_I = np.dot(np.transpose(l1_exc_spk), self.l1_exc_w)
            l1_inh_spk = self.l1_inh_lif.step(t, self.dt, l1_exc_I.flatten())
            l1_inh_I = np.dot(np.transpose(l1_inh_spk), self.l1_inh_w)
            
            # l1 -> l2 connections
            l1_l2_spk = l1_exc_spk
            l1_l2_I = self.l1_l2_syn.step(t, self.dt, l1_l2_spk, l2_exc_spk)
            
            # l2 recurrent connections
            l2_exc_spk = self.l2_exc_lif.step(t, self.dt, l1_l2_I.flatten(), l2_inh_I.flatten())
            l2_exc_I = np.dot(np.transpose(l2_exc_spk), self.l2_exc_w)
            l2_inh_spk = self.l2_inh_lif.step(t, self.dt, l2_exc_I.flatten())
            l2_inh_I = np.dot(np.transpose(l2_inh_spk), self.l2_inh_w)
            
            # update the synapses.
            # l0_l1_syn.update()
            # l1_l2_syn.update()
            
            # spike counter updates
            l1_exc_spks += l1_exc_spk
            l2_exc_spks += l2_exc_spk
            l0_spks += l0_l1_spk
        #############
        self.l0_l1_syn.reset()
        self.l1_exc_lif.reset()
        self.l1_inh_lif.reset()
        
        self.l1_l2_syn.reset()
        self.l2_exc_lif.reset()
        self.l2_inh_lif.reset()
        #############
        return l1_exc_spks, l2_exc_spks
        #############

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
        
            state = preprocess(self.env.reset())
            l1_spks, l2_spks = self.run_snn(state)
            value = np.max(l2_spks) # not sure if correct to do this ... bc choose action randomizes
            action = self.choose_action(value)
            next_state, reward, done, _ = self.env.step(action)
            
            prev_l1_spks = l1_spks
            prev_value = value
            
            i=0
            while not done:
                ###########################################     
                l1_spks, l2_spks = self.run_snn(state)
                value = np.max(l2_spks)
                _, _ = self.train_snn(prev_l1_spks, prev_value, l1_spks, value, reward)
                action = self.choose_action(l2_spks)
                next_state, reward, done, _ = self.env.step(action)
                ###########################################
                next_state = preprocess(next_state)
                state = next_state
                i += 1
                
                prev_l1_spks = l1_spks
                prev_value = value
                ###########################################
                print ('----------')
                print (e,                             \
                       np.max(l1_spks.flatten()),     \
                       np.min(l1_spks.flatten()),     \
                       np.sum(l1_spks.flatten()),     \
                       np.average(l1_spks.flatten()), \
                       np.std(l1_spks.flatten()))
                       
                print (e,                             \
                       np.max(l2_spks.flatten()),     \
                       np.min(l2_spks.flatten()),     \
                       np.sum(l2_spks.flatten()),     \
                       np.average(l2_spks.flatten()), \
                       np.std(l2_spks.flatten()))
                ###########################################
                
            # decay epsilon
            self.decay_epsilon()
                
            # update scores and print them out.
            scores.append(i)
            mean_score = np.mean(scores)
            print (mean_score)
            

agent = Solver(n_episodes=100)
agent.run()
    
    
    
    
