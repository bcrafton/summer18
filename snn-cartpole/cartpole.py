
import gym
import math
import numpy as np
import argparse
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

class Solver():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.alpha = 1e-4
        self.alpha_decay = self.alpha
        self.lmda = 0.9
        self.n_episodes = 100000
        self.n_win_ticks = 195
        self.max_env_steps = 200

        # layer sizes
        self.LAYER0 = 8
        self.LAYER1 = 48
        self.LAYER2 = 2
        
        # run time specs
        self.dt = 0.5e-3

        self.active_T = 0.35
        self.active_steps = int(self.active_T / self.dt)
        self.active_Ts = np.linspace(0, self.active_T, self.active_steps)

        self.rest_T = 0.15
        self.rest_steps = int(self.rest_T / self.dt)
        self.rest_Ts = np.linspace(self.active_T, self.active_T + self.rest_T, self.rest_steps)
        
        # initial weights
        self.l0_l1_w = np.random.normal(60.0, 5.0, size=(self.LAYER0, self.LAYER1))
        self.l1_exc_w = np.random.normal(60.0, 5.0, size=(self.LAYER1, self.LAYER1))
        self.l1_inh_w = np.random.normal(60.0, 5.0, size=(self.LAYER1, self.LAYER1))

        self.l1_l2_w = np.random.normal(60.0, 5.0, size=(self.LAYER1, self.LAYER2))
        self.l2_exc_w = np.random.normal(60.0, 5.0, size=(self.LAYER2, self.LAYER2))
        self.l2_inh_w = np.random.normal(60.0, 5.0, size=(self.LAYER2, self.LAYER2))

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
        
    def decay_epsilon(self, score):
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
        l1_exc_spks = np.zeros(shape=(self.LAYER1))
        l2_exc_spks = np.zeros(shape=(self.LAYER2))
    
        #fix me
        input_intensity = 64.0
        #############
        for s in range(self.active_steps):
            t = self.active_Ts[s]
            
            rates = state * input_intensity

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
        #############
        for s in range(self.rest_steps):
            t = self.rest_Ts[s]
            
            # l0 -> l1 connections
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
            
            # spike counter updates
            l1_exc_spks += l1_exc_spk
            l2_exc_spks += l2_exc_spk
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
            done = False
            i = 0
            
            while not done:
                l1_spks, l2_spks = self.run_snn(state)
                action = self.choose_action(l2_spks)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                state = next_state
                i += 1
                
            print (e, l1_spks.flatten(), l2_spks.flatten())
                
            scores.append(i)
            mean_score = np.mean(scores)
            self.decay_epsilon(mean_score)

if __name__ == '__main__':
    agent = Solver()
    agent.run()
    
    
    
    
    
    
    
    
    
