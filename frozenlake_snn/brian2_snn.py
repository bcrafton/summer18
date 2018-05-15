
import random
import gym
import math
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

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

reward = 0

np.random.seed(0)
num_examples = 1000
update_interval = num_examples

single_example_time =   0.35 * b2.second
resting_time = 0.15 * b2.second

v_rest_e = -65. * b2.mV 
v_rest_i = -60. * b2.mV 
v_reset_e = -65. * b2.mV
v_reset_i = -45. * b2.mV
v_thresh_e = -52. * b2.mV
v_thresh_i = -40. * b2.mV
refrac_e = 5. * b2.ms
refrac_i = 2. * b2.ms

weight = {}
weight['ee_input'] = 78.

delay = {}
delay['ee_input'] = (0*ms,10*ms)
delay['ei_input'] = (0*ms,5*ms)

NUM_LAYERS = 1

input_intensity = 2.
start_input_intensity = input_intensity

tc_pre_ee = 20*b2.ms
tc_post_1_ee = 20*b2.ms
tc_post_2_ee = 40*b2.ms
nu_ee_pre =  0.0001      # learning rate
nu_ee_post = 0.01       # learning rate
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4

tc_theta = 1e7 * b2.ms
theta_plus_e = 0.05 * b2.mV
scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
offset = 20.0*b2.mV
v_thresh_e_str = '(v>(theta - offset + v_thresh_e)) and (timer>refrac_e)'
v_thresh_i_str = 'v>v_thresh_i'
v_reset_i_str = 'v=v_reset_i'

neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
neuron_eqs_e += '\n  dtimer/dt = 0.1  : second'

neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
eqs_stdp_ee = '''
                post2before                            : 1
                dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
            '''
eqs_stdp_pre_ee = 'pre = 1.; w = clip(w + nu_ee_pre * post1 * reward, 0, wmax_ee)'
eqs_stdp_post_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before * reward, 0, wmax_ee); post1 = 1.; post2 = 1.'

#------------------------------------------------------------------------------ 
# create networks
#------------------------------------------------------------------------------ 

b2.ion()
neuron_groups = {}
input_groups = {}
connections = {}
rate_monitors = {}
spike_counters = {}

neuron_groups['e'] = b2.NeuronGroup(64*NUM_LAYERS, neuron_eqs_e, threshold=v_thresh_e_str, refractory=refrac_e, reset=scr_e, method='euler')
neuron_groups['i'] = b2.NeuronGroup(64*NUM_LAYERS, neuron_eqs_i, threshold=v_thresh_i_str, refractory=refrac_i, reset=v_reset_i_str, method='euler')

#------------------------------------------------------------------------------ 
# create network population and recurrent connections
#------------------------------------------------------------------------------ 

neuron_groups['Ae'] = neuron_groups['e'][0:64]
neuron_groups['Ai'] = neuron_groups['i'][0:64]

neuron_groups['Ae'].v = v_rest_e - 40. * b2.mV
neuron_groups['Ai'].v = v_rest_i - 40. * b2.mV
neuron_groups['e'].theta = np.ones((64)) * 20.0 * b2.mV

connName = 'AeAi'
weightMatrix = np.load('./init/AeAi.npy')
model = 'w : 1'
pre = 'ge_post += w'
post = ''
connections[connName] = b2.Synapses(neuron_groups['Ae'], neuron_groups['Ai'], model=model, on_pre=pre, on_post=post)
connections[connName].connect(True)
connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]

connName = 'AiAe'
weightMatrix = np.load('./init/AiAe.npy')
model = 'w : 1'
pre = 'gi_post += w'
post = ''
connections[connName] = b2.Synapses(neuron_groups['Ai'], neuron_groups['Ae'], model=model, on_pre=pre, on_post=post)
connections[connName].connect(True)
connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]

rate_monitors['Ae'] = b2.PopulationRateMonitor(neuron_groups['Ae'])
rate_monitors['Ai'] = b2.PopulationRateMonitor(neuron_groups['Ai'])
spike_counters['Ae'] = b2.SpikeMonitor(neuron_groups['Ae'])


#------------------------------------------------------------------------------ 
# create input population and connections from input populations 
#------------------------------------------------------------------------------ 

input_groups['Xe'] = b2.PoissonGroup(28*28, 0*Hz)
rate_monitors['Xe'] = b2.PopulationRateMonitor(input_groups['Xe'])

connName = 'XeAe'
weightMatrix = np.load('./init/XeAe.npy')

model = 'w : 1'
pre = 'ge_post += w'
post = ''
model += eqs_stdp_ee
pre += '; ' + eqs_stdp_pre_ee
post = eqs_stdp_post_ee

minDelay = delay['ee_input'][0]
maxDelay = delay['ee_input'][1]
deltaDelay = maxDelay - minDelay

connections[connName] = b2.Synapses(input_groups['Xe'], neuron_groups['Ae'], model=model, on_pre=pre, on_post=post)
connections[connName].connect(True)
connections[connName].delay = 'minDelay + rand() * deltaDelay'
connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 

class SNN:

  def __init__(self, neuron_groups, input_groups, connections, rate_monitors, spike_counters):

    self.neuron_groups = neuron_groups
    self.input_groups = input_groups
    self.connections = connections
    self.rate_monitors = rate_monitors
    self.spike_counters = spike_counters

    self.net = Network()
    for obj_list in [self.neuron_groups, self.input_groups, self.connections, self.rate_monitors, self.spike_counters]:
        for key in obj_list:
            self.net.add(obj_list[key])

    self.previous_spike_count = np.zeros(64)
    self.assignments = np.ones(64) * -1
    self.memory = deque(maxlen=1000)

    ### init run

    self.input_groups['Xe'].rates = 0 * Hz
    self.net.run(0*second)

def get_action(spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10

    for i in xrange(10):
        num_assignments[i] = len(np.where(self.assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[self.assignments == i]) / num_assignments[i]

    ret = np.argsort(summed_rates)[::-1]
    return ret

  def set_assignments(self):
      maximum_rate = [0] * 64    

      for tup in self.memory:
        for neuron in range(64):
          

        if num_inputs > 0:
          rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_inputs
        
        for i in xrange(n_e):
          if rate[i] > maximum_rate[i]:
            maximum_rate[i] = rate[i]
            assignments[i] = j 


  def infer(self, rates):
    current_spike_count = np.zeros(64)
    input_intensity = start_input_intensity

    while np.sum(current_spike_count) < 5:
      self.input_groups['Xe'].rates = rates * input_intensity * Hz
      self.net.run(single_example_time, report='text')

      current_spike_count = np.asarray(self.spike_counters['Ae'].count[:]) - self.previous_spike_count
      self.previous_spike_count = np.copy(self.spike_counters['Ae'].count[:])

      input_intensity += 1

    self.result_monitor.push_back(current_spike_count)
    self.input_numbers.push_back(current_spike_count)

    self.input_groups['Xe'].rates = 0*Hz
    self.net.run(resting_time)

  def train(self, rates):
    self.normalize()

    current_spike_count = np.zeros(64)
    input_intensity = start_input_intensity

    while np.sum(current_spike_count) < 5:
      self.input_groups['Xe'].rates = rates * input_intensity * Hz
      self.net.run(single_example_time, report='text')

      current_spike_count = np.asarray(self.spike_counters['Ae'].count[:]) - self.previous_spike_count
      self.previous_spike_count = np.copy(self.spike_counters['Ae'].count[:])

      input_intensity += 1

    self.input_groups['Xe'].rates = 0*Hz
    self.net.run(resting_time)

  def remember(self):
    self.memory.append((state, action, reward, next_state, done))

  def save(self):
    connMatrix = connections['XeAe'].w
    np.save('XeAe', connMatrix)

    np.save('theta_A', neuron_groups['Ae'].theta)

  def normalize(self):
    len_source = len(self.connections['XeAe'].source)
    len_target = len(self.connections['XeAe'].target)
    connection = np.zeros((len_source, len_target))
    connection[self.connections['XeAe'].i, self.connections['XeAe'].j] = self.connections['XeAe'].w
    temp_conn = np.copy(connection)
    colSums = np.sum(temp_conn, axis = 0)
    colFactors = weight['ee_input']/colSums
    for j in xrange(64):
        temp_conn[:,j] *= colFactors[j]
    self.connections['XeAe'].w = temp_conn[self.connections['XeAe'].i, self.connections['XeAe'].j]


snn = SNN(neuron_groups=neuron_groups, input_groups=input_groups, connections=connections, rate_monitors=rate_monitors, spike_counters=spike_counters)

env = gym.make('FrozenLake-v0')
env.reset()

num_examples = 1000
for ii in range(num_examples):

  for step in range(env.spec.timestep_limit):

    action = snn.infer(state)
    next_state, reward, done, _ = env.step(action)

    snn.remember(state, action, reward, next_state, done)

    snn.train(reward)
    reward = 0

    if done:
      break

snn.save()







