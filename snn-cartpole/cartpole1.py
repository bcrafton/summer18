
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
        
        self.ge = np.zeros(shape=(self.N))
        self.gi = np.zeros(shape=(self.N))
        self.v = np.ones(shape=(self.N)) * self.vreset
        self.last_spk = np.ones(shape=(self.N)) * -1
        
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
        self.ge = np.zeros(shape=(self.N))
        self.gi = np.zeros(shape=(self.N))
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

class RSTDP_group:
    def __init__(self, N, M, w, stdp, tc_pre_ee, tc_post_1_ee, tc_post_2_ee, nu_ee_pre, nu_ee_post, wmax_ee, lmda, gamma, alpha, e_tau):
    
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
        
        # rstdp stuff
        self.lmda = lmda
        self.gamma = gamma
        self.alpha = alpha
        self.e_tau = e_tau
        
        # synapse level variables
        self.w = w
        self.e = np.zeros(shape=(N, M))
        
        # pre level variables
        self.last_pre = np.ones(self.N) * -1
        
        # post level variables
        self.last_post = np.ones(self.M) * -1
        
        # want to store the e's so i can plot them...
        # self.es = []
        
    def step(self, t, dt, pre_spk, post_spk):
        I = np.dot(np.transpose(pre_spk), self.w)

        if self.stdp:
            de = np.zeros(shape=(self.N, self.M))
            got_pre = np.any(pre_spk)
            got_post = np.any(post_spk)
        
            if (got_pre):
                npre_spk = pre_spk == 0
                self.last_pre = self.last_pre * npre_spk
                self.last_pre += pre_spk * t
            
                post1 = np.exp(-(t - self.last_post) / self.tc_post_1_ee)
                # dont do anything to the eligibility during a pre.
                # de += -self.nu_ee_pre * np.dot(pre_spk.reshape(self.N, 1), post1.reshape(1, self.M))

            if (got_post):
                npost_spk = post_spk == 0
                self.last_post = self.last_post * npost_spk
                self.last_post += post_spk * t
                
                pre = np.exp(-(t - self.last_pre) / self.tc_pre_ee)
                post2 = np.exp(-(t - self.last_post) / self.tc_post_2_ee)
                de += self.nu_ee_post * np.dot(pre.reshape(self.N, 1), post2.reshape(1, self.M) * post_spk.reshape(1, self.M))
                
            de += -self.e / self.e_tau * dt
            self.e = np.clip(self.e + de, 0, 1.0)
            # self.es.append(self.e)
            
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
                
            self.e = np.zeros(shape=(self.N, self.M))
                
    def update(self, prev_state, prev_value, state, value, reward):
        d = reward + (self.gamma * value) - prev_value
        
        # print( np.shape(self.e), np.shape(prev_state), prev_value, np.shape(state), value, reward)
        
        dw = (self.alpha * (d + value - prev_value) * self.e) - (self.alpha * (value - prev_value) * prev_state)
        self.w += dw
        
#############

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
        self.LAYER0 = 8
        self.LAYER1 = 128
        self.LAYER2 = 2
        
        # run time specs
        self.dt = 1e-4

        ### NO LONGER USING THESE###
        self.active_T = 0.35
        self.active_steps = int(self.active_T / self.dt)
        self.active_Ts = np.linspace(0, self.active_T, self.active_steps)

        self.rest_T = 0.15
        self.rest_steps = int(self.rest_T / self.dt)
        self.rest_Ts = np.linspace(self.active_T, self.active_T + self.rest_T, self.rest_steps)
        ### NO LONGER USING THESE###
        
        # run time metrics
        self.time_elapsed = 0.0
        self.total_steps = 0.0
        
        # initial weights
        self.l0_l1_w = np.absolute(np.random.normal(0.5, 0.3, size=(self.LAYER0, self.LAYER1)))
        self.l1_exc_w = np.random.normal(0.125, 0.05, size=(self.LAYER1, self.LAYER1))
        self.l1_inh_w = np.random.normal(100.0, 0.00, size=(self.LAYER1, self.LAYER1))

        self.l1_l2_w =  np.absolute(np.random.normal(0.5, 0.3, size=(self.LAYER1, self.LAYER2)))
        self.l2_exc_w = np.random.normal(0.125, 0.05, size=(self.LAYER2, self.LAYER2))
        self.l2_inh_w = np.random.normal(100.0, 0.00, size=(self.LAYER2, self.LAYER2))

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
                            vthr=-20e-3 - 52e-3,     \
                            vrest=-65e-3,            \
                            vreset=-65e-3,           \
                            refrac_per=5e-3,         \
                            i_offset=-100e-3,        \
                            tc_theta=1e7*1e-3,       \
                            theta_plus_e=0.05e-3)

        self.l1_inh_lif = LIF_group(N=self.LAYER1,    \
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
                            
        ##########################################################
                            
        self.l1_l2_syn = RSTDP_group(N=self.LAYER1,   \
                            M=self.LAYER2,            \
                            w=self.l1_l2_w,           \
                            stdp=True,                \
                            tc_pre_ee=20e-3,          \
                            tc_post_1_ee=20e-3,       \
                            tc_post_2_ee=40e-3,       \
                            nu_ee_pre=1e-4,           \
                            nu_ee_post=1e-2,          \
                            wmax_ee=1.0,              \
                            lmda=0.9,                 \
                            gamma=0.99,               \
                            alpha=1e-6,               \
                            e_tau=2.0)

        self.l2_exc_lif = LIF_group(N=self.LAYER2,   \
                            adapt=args.train,        \
                            tau=1e-1,                \
                            theta=self.l2_theta,     \
                            vthr=-20e-3 - 52e-3,     \
                            vrest=-65e-3,            \
                            vreset=-65e-3,           \
                            refrac_per=5e-3,         \
                            i_offset=-100e-3,        \
                            tc_theta=1e7*1e-3,       \
                            theta_plus_e=0.05e-3)

        self.l2_inh_lif = LIF_group(N=self.LAYER2,    \
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

        ##########################################################

    def preprocess_state(self, state):
        assert(np.shape(state) == (4,))
        ret = np.zeros(8)
        
        ## 0, 1 < 0
        if state[0] >= 0:
          ret[0] = np.absolute(state[0])
          ret[1] = 0
        else:
          ret[0] = 0
          ret[1] = np.absolute(state[0])
          
        ## 2, 3 < 1
        if state[1] >= 0:
          ret[2] = np.absolute(state[1])
          ret[3] = 0
        else:
          ret[2] = 0
          ret[3] = np.absolute(state[1])

        ## 4, 5 < 2
        if state[2] >= 0:
          ret[4] = np.absolute(state[2])
          ret[5] = 0
        else:
          ret[4] = 0
          ret[5] = np.absolute(state[2])
        
        ## 6, 7 < 3
        if state[3] >= 0:
          ret[6] = np.absolute(state[3])
          ret[7] = 0
        else:
          ret[6] = 0
          ret[7] = np.absolute(state[3])
        
        return ret.reshape(8,1)
        
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
        l0_spks = np.zeros(shape=(self.LAYER0, 1))
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
            l0_l1_spk = np.random.rand(self.LAYER0, 1) < rates * self.dt
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
        
        # intensity, could probably make this smarter
        input_intensity = 64.0
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
            
            tmp = np.copy(l1_l2_spk).reshape(128, 1)
            self.l1_l2_syn.update(tmp, prev_value, tmp, value, reward)
            
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
        
            state = self.preprocess_state(self.env.reset())
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
                value = np.max(l2_spks) # not sure if correct to do this ... bc choose action randomizes
                _, _ = self.train_snn(prev_l1_spks, prev_value, l1_spks, value, reward)
                action = self.choose_action(l2_spks)
                next_state, reward, done, _ = self.env.step(action)
                ###########################################
                next_state = self.preprocess_state(next_state)
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

'''
T = agent.n_episodes * (agent.active_T + agent.rest_T)
steps = agent.n_episodes * (agent.active_steps + agent.rest_steps)
Ts = np.linspace(0.0, T, steps)
'''
'''
Ts = np.linspace(0.0, agent.time_elapsed, agent.total_steps)

# this annoing as hell.
# es = steps, N, M
# so you need to slice it right.
es = agent.l1_l2_syn.es
es = np.array(es)[:, 0:10, 0]

plt.plot(Ts, es)
plt.show()
''' 
    
    
    
    
