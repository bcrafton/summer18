
import random
import math
from collections import deque

import numpy as np
import matplotlib.cm as cmap
import time
import os.path
import scipy
from struct import unpack
from brian2 import *
import brian2 as b2
from brian2tools import *
import gzip

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

np.random.seed(0)
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
gamma = 0.025 * (w_max - w_min) * b2.mV

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

input = b2.PoissonGroup(16, 0*Hz)
#hidden = b2.NeuronGroup(64, lif.equ, threshold=lif.threshold, reset=lif.reset)
output = b2.NeuronGroup(4, lif.equ, threshold=lif.threshold, reset=lif.reset)

#------------------------------------------------------------------------------ 
# create input population and connections from input populations 
#------------------------------------------------------------------------------

def set_state(state):
    ret = np.zeros(16)
    for i in range(16):
        if state == i:
            ret[i] = 1
    return np.reshape(ret, [1, 16])

class GapRL:
    def __init__(self, sigma, tau_z, tau_i, gamma, w_min, w_max, beta_sigma):
        self.model = '''
                     sig = sigma(v_post, dt) : 1 (constant over dt)
                     w : volt
                     dz/dt = -z/tau_z : 1/volt (clock-driven)
                     prevSpike : second
                     '''

        self.on_pre = '''
                      v_post += w
                      prevSpike = t
                      w += gamma * reward * z 
                      '''

        self.on_post = '''
                       z += beta_sigma * exp( -(t - prevSpike) / tau_i)
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

io_syn.w = np.random.uniform(w_min, w_max, size=(64)) * 1000 * b2.volt
# io_syn.w = np.random.normal(0.026, 0.01, size=(64)) * b2.volt
# io_syn.w = weights * b2.volt

counter = SpikeMonitor(output)
previous_spike_count = np.zeros(4)

net = Network()
net.add(input)
# net.add(hidden)
net.add(output)
# net.add(ih_syn)
# net.add(ho_syn)
net.add(io_syn)
net.add(counter)

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 

env = Env()
env.reset()

start = True

wins = 0
scores = deque(maxlen=100)
num_examples = 1000
for ii in range(num_examples):

  reward = 0
  state = env.reset()
  state = set_state(state)

  done = False
  step_count = 0

  prev = io_syn.w * 1000
  while (done == False):

    if start:
      action = 0
      start = False
    else:
      input.rates = state * 128 * Hz
      # net.run(single_example_time, report='text')
      net.run(single_example_time)

      current_spike_count = np.asarray(counter.count[:]) - previous_spike_count
      previous_spike_count = np.copy(counter.count[:])
      action = np.argmax(current_spike_count)

      print (current_spike_count)

    next_state, reward, done = env.step(action)

    print (str(step_count) + "/" + str(20) + " " + str(action) + " " + str(next_state))
    step_count = step_count + 1

    # net.run(resting_time, report='text')
    net.run(resting_time)

    if done:
      if (reward > 0):
        wins = wins + 1
      scores.append(reward > 0)
      mean_score = np.mean(scores)
      print (mean_score, wins)
      print (io_syn.w * 1000 - prev)

      np.save("snn_weights", io_syn.w)

    state = next_state
    state = set_state(state)

    reward = 0







