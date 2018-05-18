
import random
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import time
import brian2 as b2
from brian2tools import *

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

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 

class Env():
    def __init__(self):
        self.state = 0
        self.steps = 0

        self.nrows = 4
        self.ncols = 4

        self.grid = np.ones(self.nrows * self.ncols) * -5
        self.wins = [15]
        self.fails = [10]

        self.left = []
        self.right = []
        self.up = []
        self.down = []

        for i in self.wins:
            self.grid[i] = 100

        for i in self.fails:
            self.grid[i] = -100

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
            reward = 100
            done = True
        elif next_state in self.fails:
            reward = -100
            done = True
        else:
            reward = -5
            if (self.steps >= 20):
                done = True
            else:
                done = False

        self.state = next_state
        return next_state, reward, done

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 

num_examples = 1000
update_interval = num_examples

single_example_time =   0.35 * b2.second
resting_time = 0.15 * b2.second

NUM_LAYERS = 1

input_intensity = 2.
start_input_intensity = input_intensity

#------------------------------------------------------------------------------ 
# create networks
#------------------------------------------------------------------------------ 

b2.ion()

tau_i = 20 * b2.ms  # Time constant for LIF leak
v_r = 0 * b2.mV  # Reset potential
th_i = 16 * b2.mV  # Threshold potential
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
#hidden = b2.NeuronGroup(64, lif.equ, threshold=lif.threshold, reset=lif.reset)
output = b2.NeuronGroup(4, lif.equ, threshold=lif.threshold, reset=lif.reset)

#------------------------------------------------------------------------------ 
# create input population and connections from input populations 
#------------------------------------------------------------------------------

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

class GapRL:
    def __init__(self, sigma, tau_z, tau_i, gamma, w_min, w_max, beta_sigma):
        self.model = '''
                     w : volt
                     prevSpike : second
                     '''

        self.on_pre = '''
                      v_post += w
                      prevSpike = t
                      '''

        self.on_post = '''
                       '''

syn = GapRL(sigma, tau_z, tau_i, gamma, w_min, w_max, beta_sigma)

# ih_syn = b2.Synapses(input, hidden, model=syn.model, on_pre=syn.on_pre, on_post=syn.on_post)
# ho_syn = b2.Synapses(hidden, output, model=syn.model, on_pre=syn.on_pre, on_post=syn.on_post)

io_syn = b2.Synapses(input, output, model=syn.model, on_pre=syn.on_pre, on_post=syn.on_post)
io_syn.connect(True)

'''
weights = np.load("weights.npy")
weights = weights[0]
weights = weights.flatten()
avg = np.average(weights)
weights = weights * (0.026 / np.absolute(avg))
'''
# weights = np.load("snn_weights_1150.npy")

io_syn.w = np.ones(64) * 0.25 * b2.volt
# io_syn.w = np.random.uniform(w_min, w_max, size=(64)) * 1000 * b2.volt
# io_syn.w = np.random.normal(0.026, 0.01, size=(64)) * b2.volt
# io_syn.w = weights * b2.volt

counter = b2.SpikeMonitor(output)
previous_spike_count = np.zeros(4)

net = b2.Network()
net.add(input)
# net.add(hidden)
net.add(output)
# net.add(ih_syn)
# net.add(ho_syn)
net.add(io_syn)
net.add(counter)

class Solver():
    def __init__(self, n_episodes=1000, max_env_steps=None, gamma=1.0, epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.99, alpha=0.04, alpha_decay=0.04, batch_size=32, quiet=False):
        self.memory = deque(maxlen=64)

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

        self.model = Sequential()
        self.model.add(Dense(4, input_shape=(16,), activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    '''
    def preprocess_state(self, state):
        ret = np.zeros(16)
        for i in range(16):
            if state == i:
                ret[i] = 1
        return np.reshape(ret, [1, 16])
    '''

    def replay(self, batch_size):
        x_batch, y_batch = [], []

        # wow that was frusterating.
        # from brian2 import *
        # breaks all calls to random.
        # wasted a bunch of time
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        # idx = np.random.choice( len(self.memory), self.batch_size )
        # minibatch = self.memory[idx]

        # sz = min(self.batch_size, len(self.memory))
        # minibatch = self.memory[:sz]

        # minibatch = self.memory
        # minibatch = list(minibatch)

        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state))
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

    def run(self):
      start = True

      previous_spike_count = np.zeros(4)

      wins = 0
      scores = deque(maxlen=100)
      num_examples = 1000

      prev = io_syn.w * 1000

      for ii in range(num_examples):

        reward = 0
        state = self.env.reset()
        state = num_to_state(state)

        done = False
        step_count = 0

        prev = io_syn.w * 1000
        while (done == False):
          step_count = step_count + 1

          if start:
            action = 0
            start = False
          else:
            input.rates = state * 128 * b2.Hz
            net.run(single_example_time)

            current_spike_count = np.asarray(counter.count[:]) - previous_spike_count
            previous_spike_count = np.copy(counter.count[:])
            action = np.argmax(current_spike_count)

            print (current_spike_count)

          next_state, reward, done = self.env.step(action)
          next_state = num_to_state(next_state)

          self.remember(state, action, reward, next_state, done)          

          print (str(step_count) + "/" + str(20) + " " + str(action) + " " + str(state_to_num(next_state)))

          net.run(resting_time)

          if done:
            if (reward > 0):
              wins = wins + 1

            scores.append(reward > 0)
            mean_score = np.mean(scores)
            print (mean_score, wins)

            weights = self.model.get_weights()[0].flatten()
            weights = weights * ((w_max-w_min) / np.absolute(np.average(weights)))
            weights = weights + np.absolute(np.min(weights))
            # weights = weights * b2.volt

            # io_syn.w = self.model.get_weights()[0].flatten() * b2.volt
            io_syn.w = weights

            disp_norm (io_syn.w * 1000 - prev)
            disp (io_syn.w)

            np.save("snn_weights", io_syn.w)

            self.replay(self.batch_size)

          state = next_state
          reward = 0        

if __name__ == '__main__':
    agent = Solver()
    agent.run()




