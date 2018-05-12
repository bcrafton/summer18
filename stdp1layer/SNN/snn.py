
import numpy as np
import matplotlib.cm as cmap
import time
import os.path
import scipy 
import cPickle as pickle

from struct import unpack
from brian import *
import gzip

########### INIT STUFF ###########
np.random.seed(0)

set_global_preferences( 
  defaultclock = Clock(dt=0.5*ms), # The default clock to use if none is provided or defined in any enclosing scope.
  useweave = False, # Defines whether or not functions should use inlined compiled C code where defined.
  gcc_options = ['-ffast-math -march=native'],  # Defines the compiler switches passed to the gcc compiler. 
  #For gcc versions 4.2+ we recommend using -march=native. By default, the -ffast-math optimizations are turned on 
  usecodegen = True,  # Whether or not to use experimental code generation support.
  usecodegenweave = True,  # Whether or not to use C with experimental code generation support.
  usecodegenstateupdate = True,  # Whether or not to use experimental code generation support on state updaters.
  usecodegenthreshold = False,  # Whether or not to use experimental code generation support on thresholds.
  usenewpropagate = True,  # Whether or not to use experimental new C propagation functions.
  usecstdp = True,  # Whether or not to use experimental new C STDP.
) 

########### INIT STUFF ###########

single_example_time =   0.35 * second
resting_time = 0.15 * second

v_rest_e = -65. * mV 
v_rest_i = -60. * mV 
v_reset_e = -65. * mV
v_reset_i = -45. * mV
v_thresh_e = -52. * mV
v_thresh_i = -40. * mV
refrac_e = 5. * ms
refrac_i = 2. * ms

conn_structure = 'dense'

weight = {}
weight['ee_input'] = 78.

delay = {}
delay['ee_input'] = (0*ms,10*ms)
delay['ei_input'] = (0*ms,5*ms)

NUM_LAYERS = 1

input_intensity = 2.
start_input_intensity = input_intensity

tc_pre_ee = 20*ms
tc_post_1_ee = 20*ms
tc_post_2_ee = 40*ms
nu_ee_pre =  0.0001      # learning rate
nu_ee_post = 0.01       # learning rate
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4

tc_theta = 1e7 * ms
theta_plus_e = 0.05 * mV
scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
offset = 20.0*mV
v_thresh_e = '(v>(theta - offset + -52. * mV)) * (timer>refrac_e)'


neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
neuron_eqs_e += '\n  dtimer/dt = 100.0  : ms'

neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
eqs_stdp_ee = '''
                post2before                            : 1.0
                dpre/dt   =   -pre/(tc_pre_ee)         : 1.0
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1.0
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1.0
            '''
eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post1'
eqs_stdp_post_ee = 'post2before = post2; w += nu_ee_post * pre * post2before; post1 = 1.; post2 = 1.'

#------------------------------------------------------------------------------ 
# create networks
#------------------------------------------------------------------------------ 

neuron_groups = {}
input_groups = {}
connections = {}
stdp_methods = {}
rate_monitors = {}
spike_counters = {}

neuron_groups['e'] = NeuronGroup(400 * NUM_LAYERS, neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= scr_e, compile = True, freeze = True)
neuron_groups['i'] = NeuronGroup(400 * NUM_LAYERS, neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i, compile = True, freeze = True)

#------------------------------------------------------------------------------ 
# create network population and recurrent connections
#------------------------------------------------------------------------------ 

neuron_groups['Ae'] = neuron_groups['e'].subgroup(400)
neuron_groups['Ai'] = neuron_groups['i'].subgroup(400)

neuron_groups['Ae'].v = v_rest_e - 40. * mV
neuron_groups['Ai'].v = v_rest_i - 40. * mV
neuron_groups['e'].theta = np.ones((400)) * 20.0*mV

connName = 'AeAi'
weightMatrix = np.load('./init/AeAi.npy')
connections[connName] = Connection(neuron_groups['Ae'], neuron_groups['Ai'], structure=conn_structure, state = 'g'+'e')
connections[connName].connect(neuron_groups['Ae'], neuron_groups['Ai'], weightMatrix)

connName = 'AiAe'
weightMatrix = np.load('./init/AiAe.npy')
connections[connName] = Connection(neuron_groups['Ai'], neuron_groups['Ae'], structure=conn_structure, state = 'g'+'i')
connections[connName].connect(neuron_groups['Ai'], neuron_groups['Ae'], weightMatrix)

rate_monitors['Ae'] = PopulationRateMonitor(neuron_groups['Ae'], bin = (single_example_time+resting_time)/second)
rate_monitors['Ai'] = PopulationRateMonitor(neuron_groups['Ai'], bin = (single_example_time+resting_time)/second)
spike_counters['Ae'] = SpikeCounter(neuron_groups['Ae'])


#------------------------------------------------------------------------------ 
# create input population and connections from input populations 
#------------------------------------------------------------------------------ 

input_groups['Xe'] = PoissonGroup(28*28, 0)
rate_monitors['Xe'] = PopulationRateMonitor(input_groups['Xe'], bin = (single_example_time + resting_time) / second)

connName = 'XeAe'
weightMatrix = np.load('./init/XeAe.npy')
connections[connName] = Connection(input_groups['Xe'], neuron_groups['Ae'], structure=conn_structure, state = 'g' + 'e', delay=True, max_delay=delay['ee_input'][1])
connections[connName].connect(input_groups['Xe'], neuron_groups['Ae'], weightMatrix, delay=delay['ee_input'])

stdp_methods['XeAe'] = STDP(connections['XeAe'], eqs=eqs_stdp_ee, pre = eqs_stdp_pre_ee, post = eqs_stdp_post_ee, wmin=0., wmax= wmax_ee)

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 

class SNN:

  def __init__(self, neuron_groups, input_groups, connections, stdp_methods, rate_monitors, spike_counters):

    self.neuron_groups = neuron_groups
    self.input_groups = input_groups
    self.connections = connections
    self.stdp_methods = stdp_methods
    self.rate_monitors = rate_monitors
    self.spike_counters = spike_counters
    
    self.previous_spike_count = np.zeros(400)

    run(0*second)

    print "Inst SNN"

  def train(self, rates):
    self.normalize()

    current_spike_count = np.zeros(400)
    input_intensity = start_input_intensity

    while np.sum(current_spike_count) < 5:
      self.input_groups['Xe'].rate = rates * input_intensity
      run(single_example_time, report='text')

      current_spike_count = np.asarray(self.spike_counters['Ae'].count[:]) - self.previous_spike_count
      self.previous_spike_count = np.copy(self.spike_counters['Ae'].count[:])

      input_intensity += 1

    self.input_groups['Xe'].rate = 0
    run(resting_time)

  def save(self):
    connMatrix = self.connections['XeAe'][:]
    np.save('XeAe', connMatrix)
    np.save('theta_A', self.neuron_groups['Ae'].theta)

  def normalize(self):
    connection = connections['XeAe'][:]
    temp_conn = np.copy(connection)
    colSums = np.sum(temp_conn, axis = 0)
    colFactors = weight['ee_input']/colSums
    for j in xrange(400):
      connection[:, j] *= colFactors[j]


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


load_data()
snn = SNN(neuron_groups, input_groups, connections, stdp_methods, rate_monitors, spike_counters)

num_examples = 1000
for j in range(num_examples):
  ex = training_set[j] * 32.
  snn.train(ex)

snn.save()







