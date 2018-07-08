 
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
import matplotlib.pyplot as plt

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

def normalize_weights():
    for connName in connections:
        if connName[1] == 'e' and connName[3] == 'e':
            connection = connections[connName][:]
            temp_conn = np.copy(connection)
            colSums = np.sum(temp_conn, axis = 0)
            colFactors = weight['ee_input']/colSums
            for j in xrange(n_e):#
                connection[:,j] *= colFactors[j]
    
#------------------------------------------------------------------------------ 
# load MNIST
#------------------------------------------------------------------------------

load_data()

#------------------------------------------------------------------------------ 
# set parameters and equations
#------------------------------------------------------------------------------
test_mode = True

np.random.seed(0)
num_examples = 1000 * 1
use_testing_set = True
update_interval = num_examples

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
delay = {}
input_population_names = ['X']
population_names = ['A']
input_connection_names = ['XA']
save_conns = ['XeAe']
input_conn_names = ['ee_input'] 
recurrent_conn_names = ['ei', 'ie']
weight['ee_input'] = 78.
delay['ee_input'] = (0*b2.ms,10*b2.ms)
delay['ei_input'] = (0*b2.ms,5*b2.ms)
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

scr_e = 'v = v_reset_e; timer = 0*ms'

offset = 20.0 * b2.mV
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
neuron_eqs_e += '\n  theta      :volt'
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
stdp_methods = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
result_monitor = np.zeros((update_interval, n_e))

n1 = b2.NeuronGroup(1, neuron_eqs_e, threshold= v_thresh_e_str, refractory= refrac_e, reset= scr_e, method='euler')
n1.v = v_rest_e - 40. * b2.mV
n1.theta = np.load('theta_A.npy')[0] * b2.volt
mon = b2.StateMonitor(n1, ('ge', 'v'), True)

n2 = b2.PoissonGroup(1, 5*Hz)
spk = b2.SpikeMonitor(n2)

n3 = b2.NeuronGroup(1, neuron_eqs_i, threshold= v_thresh_i_str, refractory= refrac_i, reset= v_reset_i_str, method='euler')
n3.v = v_rest_i - 40. * b2.mV

#------------------------------------------------------------------------------ 

connName = 'AeAi'
weightMatrix = np.load('AeAi.npy')
weightMatrix = weightMatrix.flatten()
model = 'w : 1'
pre = 'ge_post += w'
post = ''
syn13 = b2.Synapses(n1, n3, model=model, on_pre=pre, on_post=post)
syn13.connect(True)
syn13.w = weightMatrix[0]

connName = 'AiAe'
weightMatrix = np.load('AiAe.npy')
weightMatrix = weightMatrix.flatten()
model = 'w : 1'
pre = 'gi_post += w'
post = ''
syn31 = b2.Synapses(n3, n1, model=model, on_pre=pre, on_post=post)
syn31.connect(True)
syn31.w = weightMatrix[0]

connName = 'XeAe'
weightMatrix = np.load('XeAe.npy')
weightMatrix = weightMatrix.flatten()
model = 'w : 1'
pre = 'ge_post += w'
post = ''
minDelay = delay['ee_input'][0]
maxDelay = delay['ee_input'][1]
deltaDelay = maxDelay - minDelay
syn21 = b2.Synapses(n2, n1, model=model, on_pre=pre, on_post=post)
syn21.connect(True)
syn21.delay = 'minDelay + rand() * deltaDelay'
syn21.w = weightMatrix[0]

#------------------------------------------------------------------------------ 
# run the simulation and set inputs
#------------------------------------------------------------------------------ 

net = Network()
net.add(n1)
net.add(n2)
net.add(n3)

net.add(syn13)
net.add(syn31)
net.add(syn21)

net.add(mon)
net.add(spk)

net.run(5.0 * b2.second, report='text')

print spk.count[:]

t = np.linspace(0, 5.0, len(mon.v[0]))
plt.plot(t, mon.v[0])

#t = np.linspace(0, 5.0, len(mon.v[0]))
#plt.plot(t, mon.v[0])

plt.savefig('plt.png')



