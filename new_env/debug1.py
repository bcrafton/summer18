
import random
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.constraints import non_neg
import keras

import time
import brian2 as b2
from brian2tools import *

from operator import itemgetter

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 

def action_to_str(action):
  return '{:+05.3f}'.format(action)

def to_str(actions):
  return [action_to_str(actions[0]), action_to_str(actions[2]) + " " + action_to_str(actions[3]), action_to_str(actions[1])]

def disp_norm(x):

  x = np.asarray(x)
  x = x.reshape(16,4)
  max_val = np.max( np.absolute(x) )
  x = x / max_val
  x = np.round(x, 3)

  for i in range(4):
    grid_str = ["", "", ""]
    for j in range(4):
      state = (4 - i - 1) * 4 + j
      state_str = to_str( x[state] )

      grid_str[0] += "         " + state_str[0]
      grid_str[1] += "   " + state_str[1]
      grid_str[2] += "         " + state_str[2]

    print (grid_str[0])
    print (grid_str[1])
    print (grid_str[2])
    print ("")

def disp(x):

  x = np.asarray(x)
  x = x.reshape(16,4)
  x = np.round(x, 3)

  for i in range(4):
    grid_str = ["", "", ""]
    for j in range(4):
      state = (4 - i - 1) * 4 + j
      state_str = to_str( x[state] )

      grid_str[0] += "         " + state_str[0]
      grid_str[1] += "   " + state_str[1]
      grid_str[2] += "         " + state_str[2]

    print (grid_str[0])
    print (grid_str[1])
    print (grid_str[2])
    print ("")

def num_to_state(state):
    ret = np.zeros(16)
    for i in range(16):
        if state == i:
            ret[i] = 1
    # return np.reshape(ret, 16)
    return np.reshape(ret, [1, 16])

def state_to_num(state):
    for i in range(16):
        if state[0][i]:
            return i

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 

'''
C D E F
8 9 A B
4 5 6 7
0 1 2 3
'''

def manhatten(coord, dest):
    return abs(dest[0] - coord[0]) + abs(dest[1] - coord[1])

class Env():
    def __init__(self):
        self.state = 0
        self.steps = 0

        self.nrows = 4
        self.ncols = 4

        self.wins = [15]
        self.fails = [10]

        self.small_rewards = [(3, 14), (0, 11)]
        self.tiny_rewards = [(3, 13), (0, 7)]

        self.left = []
        self.right = []
        self.up = []
        self.down = []

        for i in range(self.nrows * self.ncols):
            if ( i % self.ncols == 3 ):
                self.right.append(i)

            if ( i % self.ncols == 0 ):
                self.left.append(i)

            if ( math.floor(i / self.ncols) == (self.nrows-1) ):
                self.up.append(i)

            if ( math.floor(i / self.ncols) == 0 ):
                self.down.append(i)

    def reset(self):
        self.state = 0
        self.steps = 0
        return self.state

    def step(self, action):
        self.steps = self.steps + 1

        next_state = self.state
        if (action == 0) and (self.state not in self.up):
            next_state = self.state + 4
        elif (action == 1) and (self.state not in self.down):
            next_state = self.state - 4
        elif (action == 2) and (self.state not in self.left):
            next_state = self.state - 1
        elif (action == 3) and (self.state not in self.right):
            next_state = self.state + 1

        if next_state in self.wins:
            reward = 1000
            done = True
        elif next_state in self.fails:
            reward = 0
            done = True
        else:
            if (action, self.state) in self.small_rewards:
                reward = 25
            elif (action, self.state) in self.tiny_rewards:
                reward = 25
            else:
                reward = 25

            if (self.steps >= 20):
                done = True
            else:
                done = False

        self.state = next_state
        return next_state, reward, done

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 

single_example_time =   0.35 * b2.second
resting_time = 0.15 * b2.second

NUM_LAYERS = 1

input_intensity = 2.
start_input_intensity = input_intensity

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 

b2.ion()

tau_i = 20 * b2.ms 
v_r = 0 * b2.mV  
th_i = 16 * b2.mV  
tau_sigma = 20 * b2.ms
beta_sigma = 0.2 / b2.mV

tau_z = 5 * b2.ms
w_min_i = -0.1 * b2.mV
w_max_i = 1.5 * b2.mV
gamma_i = 0.025 * (w_max_i - w_min_i) * b2.mV

w_min = -0.4 * b2.mV
w_max = 1 * b2.mV
gamma = 0.100 * (w_max - w_min) * b2.mV

@b2.check_units(voltage=b2.volt, dt=b2.second, result=1)
def sigma(voltage, dt):
    sv = dt / tau_sigma * b2.exp(beta_sigma * (voltage - th_i))
    sv = sv.clip(0, 1 - 1e-8)
    return sv

class LIF:
    def __init__(self, tau_i, v_r, sigma, tau_sigma, beta_sigma):
        self.equ = b2.Equations('dv/dt = -v/tau_i : volt')
        self.threshold = 'rand() < sigma(v, dt)'
        self.reset = 'v = v_r'

lif = LIF(tau_i, v_r, sigma, tau_sigma, beta_sigma)

input = b2.PoissonGroup(16, 0*b2.Hz)
output = b2.NeuronGroup(4, lif.equ, threshold=lif.threshold, reset=lif.reset)

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------

class GapRL:
    def __init__(self, sigma, tau_z, tau_i, gamma, w_min, w_max, beta_sigma):
        self.model = '''
                     w : volt
                     '''

        # on pre, increase post synaptic neuron voltage by w.
        self.on_pre = '''
                      v_post += w
                      '''

        self.on_post = '''
                       '''

syn = GapRL(sigma, tau_z, tau_i, gamma, w_min, w_max, beta_sigma)
io_syn = b2.Synapses(input, output, model=syn.model, on_pre=syn.on_pre, on_post=syn.on_post)
io_syn.connect(True)
io_syn.w = np.ones(64) * 0.25 * b2.volt
counter = b2.SpikeMonitor(output)

net = b2.Network()
net.add(input)
net.add(output)
net.add(io_syn)
net.add(counter)

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------

class Solver():
    def __init__(self, n_episodes=1000, max_env_steps=None, gamma=1.0, epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.99, alpha=1.00, alpha_decay=0.04, batch_size=32, quiet=False):
        self.memory = deque(maxlen=64)
        self.hist = {}

        self.env = Env()

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.quiet = quiet
        self.decay_step = 10
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # self.model.add(Dense(4, input_dim=16, activation='linear', use_bias=False, kernel_constraint=non_neg()))
        layer = keras.layers.Dense(4, input_dim=16, activation='linear', kernel_constraint=non_neg())
        self.model = Sequential([layer])
        # model = keras.models.Model(inputs=layer_in, outputs=layer_out)
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

        print ("the weights")
        print (self.model.get_weights())

        init_weights = [np.ones(shape=(16, 4)) * 1000.0, np.zeros(shape=(4))]
        self.model.set_weights(init_weights)

        self.previous_spike_count = np.zeros(4)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        if (np.random.random() <= epsilon):
            ret = random.randint(0, 3)
            rand = True
        else:
            ret = np.argmax(self.model.predict(state))
            rand = False

        return ret, rand

    def get_epsilon(self, t):
        return max(self.epsilon_min, self.epsilon)

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward+5 if done else reward+5 + self.gamma * np.max(self.model.predict(next_state))
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

    def spike_weights(self):
        weights = self.model.get_weights()[0]
        # weights = np.transpose(weights)
        # weights = weights.flatten()
        weights = weights - np.min(weights) + 1
        weights = weights * ((w_max-w_min) * 2 / np.absolute(np.average(weights)))
        # weights = weights + np.absolute(np.min(weights))
        return weights

    def run(self):
        scores = deque(maxlen=100)
        sum_avg_delta = 0

        for e in range(100000):
            state = self.env.reset()
            state = num_to_state(state)

            reward_sum = 0
            done = False
            step = 0
            prev = self.model.get_weights()[0]
            sum_avg_delta = sum_avg_delta + np.average(np.absolute(prev))

            if (np.mean(scores) > 0.5) and (e > 100):
                break

            while not done:
                step = step + 1
                qaction, rand = self.choose_action(state, self.get_epsilon(e))

                if rand:
                    snn_action = qaction
                else:
                    snn_action, spikes, io_syn_weights, _ = self.run_snn()

                state_num = state_to_num(state)
                diff = self.model.get_weights()[0][state_num][qaction] - self.model.get_weights()[0][state_num][snn_action]
                if (qaction != snn_action):
                    print ("diff = " + str(diff))
                    print (self.model.get_weights()[0][state_to_num(state)], io_syn_weights, spikes)

                action = snn_action
                next_state, reward, done = self.env.step(action)
                next_state = num_to_state(next_state)
                reward_sum = reward_sum + reward
                self.remember(state, action, reward, next_state, done)

                '''
                state_num = state_to_num(state)
                if (state_num, action) in self.hist:
                    self.hist[(state_num, action)] = self.hist[(state_num, action)] + 1
                else:
                    self.hist[(state_num, action)] = 1
                '''

                itr = str(e) + " "
                itr = itr + str(step) + "/" + str(20) + " "
                itr = itr + str(state_to_num(state)) + " (" + str(qaction) + ", " + str(snn_action) + ") " + str(state_to_num(next_state)) + " "
                itr = itr + str(reward) + " "
                itr = itr + str(self.epsilon) + " "
                print (itr)

                state = next_state

                if done:
                    self.replay(self.batch_size)

                    if (reward_sum > 1000) and (self.epsilon > self.epsilon_min):
                        self.epsilon *= 0.7

                    scores.append(reward_sum > 1000)
                    mean_score = np.mean(scores)

                    print (e, step, mean_score, self.epsilon, sum_avg_delta)

                    # disp (self.model.get_weights()[0] - prev)
                    disp (self.model.get_weights()[0])

                    # print (self.hist)
                    # for key in sorted(self.hist, key=itemgetter(0)):
                    #     print (key, self.hist[key])

                    break

    def run_snn(self):

        # print (io_syn[1]._indices())

        state = self.env.state
        state = num_to_state(state)

        weights = self.spike_weights()
        for ii in range(16):
            for jj in range(4):
                io_syn.w[ii, jj] = weights[ii, jj]

        tmp = io_syn.w[self.env.state, :]
        tmp = np.array(tmp)
        io_syn_weights = tmp

        # take the current spiking count here.
        input.rates = state * 1000 * b2.Hz
        net.run(single_example_time)
        current_spike_count = np.asarray(counter.count[:]) - self.previous_spike_count
        voltages = np.array(output.v / b2.volt)

        # set previous to be after resting.
        input.rates = 0 * b2.Hz
        net.run(resting_time)
        self.previous_spike_count = np.copy(counter.count[:])

        action = np.argmax(current_spike_count) 

        return action, current_spike_count, io_syn_weights, voltages

if __name__ == '__main__':
    agent = Solver()
    agent.run()




