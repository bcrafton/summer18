 
import numpy as np
import matplotlib.cm as cmap
import time
import os.path
import scipy
import cPickle as pickle
from struct import unpack
from brian2 import *
import brian2 as b2
from brian2tools import *
import gzip

#------------------------------------------------------------------------------ 
# functions
#------------------------------------------------------------------------------     
def load_data():
  global training_set, training_labels, testing_set, testing_labels
  f = gzip.open('mnist.pkl.gz', 'rb')
  train, valid, test = pickle.load(f)

  [training_set, training_labels] = train
  [validation_set, validation_labels] = valid
  [testing_set, testing_labels] = test

  for i in range( len(training_set) ):
    training_set[i] = training_set[i].reshape(28*28)

  for i in range( len(testing_set) ):
    testing_set[i] = testing_set[i].reshape(28*28)

  f.close()

def save_connections(ending = ''):
    connMatrix = connections['XeAe'].w
    np.save('XeAe', connMatrix)

def save_theta(ending = ''):
    np.save('theta_A', neuron_groups['Ae'].theta)

def normalize_weights():
    for connName in connections:
        if connName[1] == 'e' and connName[3] == 'e':
            len_source = len(connections[connName].source)
            len_target = len(connections[connName].target)
            connection = np.zeros((len_source, len_target))
            connection[connections[connName].i, connections[connName].j] = connections[connName].w
            temp_conn = np.copy(connection)
            colSums = np.sum(temp_conn, axis = 0)
            colFactors = weight['ee_input']/colSums
            for j in xrange(n_e):#
                temp_conn[:,j] *= colFactors[j]
            connections[connName].w = temp_conn[connections[connName].i, connections[connName].j]

#------------------------------------------------------------------------------ 
# load MNIST
#------------------------------------------------------------------------------

load_data()

#------------------------------------------------------------------------------ 
# set parameters and equations
#------------------------------------------------------------------------------

np.random.seed(0)
num_examples = 1000
update_interval = num_examples

ending = ''
n_input = 784
n_e = 400
n_i = n_e 

single_example_time =   0.35 * b2.second #
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
delay['ee_input'] = (0*b2.ms,10*b2.ms)
delay['ei_input'] = (0*b2.ms,5*b2.ms)

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
eqs_stdp_pre_ee = 'pre = 1.; w = clip(w + nu_ee_pre * post1, 0, wmax_ee)'
eqs_stdp_post_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'
    
b2.ion()
fig_num = 1
neuron_groups = {}
input_groups = {}
connections = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
result_monitor = np.zeros((update_interval,n_e))

neuron_groups['e'] = b2.NeuronGroup(n_e*NUM_LAYERS, neuron_eqs_e, threshold=v_thresh_e_str, refractory=refrac_e, reset=scr_e, method='euler')
neuron_groups['i'] = b2.NeuronGroup(n_i*NUM_LAYERS, neuron_eqs_i, threshold=v_thresh_i_str, refractory=refrac_i, reset=v_reset_i_str, method='euler')

#------------------------------------------------------------------------------ 
# create network population and recurrent connections
#------------------------------------------------------------------------------ 

neuron_groups['Ae'] = neuron_groups['e'][0:n_e]
neuron_groups['Ai'] = neuron_groups['i'][0:n_i]

neuron_groups['Ae'].v = v_rest_e - 40. * b2.mV
neuron_groups['Ai'].v = v_rest_i - 40. * b2.mV
neuron_groups['e'].theta = np.ones((n_e)) * 20.0 * b2.mV

connName = 'AeAi'
weightMatrix = np.load('../random/AeAi.npy')
model = 'w : 1'
pre = 'ge_post += w'
post = ''
connections[connName] = b2.Synapses(neuron_groups['Ae'], neuron_groups['Ai'], model=model, on_pre=pre, on_post=post)
connections[connName].connect(True)
connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]

connName = 'AiAe'
weightMatrix = np.load('../random/AiAe.npy')
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

input_groups['Xe'] = b2.PoissonGroup(n_input, 0*Hz)
rate_monitors['Xe'] = b2.PopulationRateMonitor(input_groups['Xe'])

connName = 'XeAe'
weightMatrix = np.load('../random/XeAe.npy')

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
# run the simulation and set inputs
#------------------------------------------------------------------------------ 

net = Network()
for obj_list in [neuron_groups, input_groups, connections, rate_monitors, spike_monitors, spike_counters]:
    for key in obj_list:
        net.add(obj_list[key])

previous_spike_count = np.zeros(n_e)
input_numbers = [0] * num_examples

input_groups['Xe'].rates = 0*Hz

net.run(0*second)

prev_weights = connections['XeAe'].w

j = 0
while j < num_examples:

    normalize_weights()
    rates = training_set[j] * 32. *  input_intensity

    input_groups['Xe'].rates = rates * Hz
    net.run(single_example_time, report='text')

    current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])

    if np.sum(current_spike_count) < 5:
        input_intensity += 1
    else:
        input_intensity = start_input_intensity
        j += 1

    input_groups['Xe'].rates = 0*Hz
    net.run(resting_time)

    print( np.sum(connections['XeAe'].w - prev_weights) )
    prev_weights = connections['XeAe'].w

#------------------------------------------------------------------------------ 
# save results
#------------------------------------------------------------------------------ 
print 'save results'
save_theta()
save_connections()




