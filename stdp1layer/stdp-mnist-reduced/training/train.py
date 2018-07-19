 
import numpy as np
import matplotlib.cm as cmap
import time
import os.path
import scipy 
import cPickle as pickle
import brian_no_units  #import it to deactivate unit checking --> This should NOT be done for testing/debugging 
import brian as b
from struct import unpack
from brian import *
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
    connMatrix = connections['XeAe'][:]
    np.save('XeAe', connMatrix)

def save_theta(ending = ''):
    np.save('theta_A', neuron_groups['Ae'].theta)

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

b.set_global_preferences( 
                        defaultclock = b.Clock(dt=0.5*b.ms), # The default clock to use if none is provided or defined in any enclosing scope.
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


np.random.seed(0)
num_examples = 10000

ending = ''
n_input = 784
n_e = 400
n_i = n_e 

single_example_time =   0.35 * b.second
resting_time = 0.15 * b.second
update_interval = num_examples

v_rest_e = -65. * b.mV 
v_rest_i = -60. * b.mV 
v_reset_e = -65. * b.mV
v_reset_i = -45. * b.mV
v_thresh_e = -52. * b.mV
v_thresh_i = -40. * b.mV
refrac_e = 5. * b.ms
refrac_i = 2. * b.ms

weight = {}
weight['ee_input'] = 78.

delay = {}
delay['ee_input'] = (0*b.ms,10*b.ms)
delay['ei_input'] = (0*b.ms,5*b.ms)

NUM_LAYERS = 1

input_intensity = 2.
start_input_intensity = input_intensity

tc_pre_ee = 20*b.ms
tc_post_1_ee = 20*b.ms
tc_post_2_ee = 40*b.ms
nu_ee_pre =  0.0001      # learning rate
nu_ee_post = 0.01       # learning rate
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4

tc_theta = 1e7 * b.ms
theta_plus_e = 0.05 * b.mV
scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
offset = 20.0*b.mV
v_thresh_e = '(v>(theta - offset + ' + str(v_thresh_e) + ')) * (timer>refrac_e)'


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
    
b.ion()
fig_num = 1
neuron_groups = {}
input_groups = {}
connections = {}
stdp_methods = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}

neuron_groups['e'] = b.NeuronGroup(n_e*NUM_LAYERS, neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= scr_e, compile = True, freeze = True)
neuron_groups['i'] = b.NeuronGroup(n_i*NUM_LAYERS, neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i, compile = True, freeze = True)

#------------------------------------------------------------------------------ 
# create network population and recurrent connections
#------------------------------------------------------------------------------ 

neuron_groups['Ae'] = neuron_groups['e'].subgroup(n_e)
neuron_groups['Ai'] = neuron_groups['i'].subgroup(n_i)

neuron_groups['Ae'].v = v_rest_e - 40. * b.mV
neuron_groups['Ai'].v = v_rest_i - 40. * b.mV
neuron_groups['e'].theta = np.ones((n_e)) * 20.0*b.mV

connName = 'AeAi'
weightMatrix = np.load('../random/AeAi.npy')
connections[connName] = b.Connection(neuron_groups['Ae'], neuron_groups['Ai'], structure='dense', state = 'g'+'e')
connections[connName].connect(neuron_groups['Ae'], neuron_groups['Ai'], weightMatrix)

connName = 'AiAe'
weightMatrix = np.load('../random/AiAe.npy')
connections[connName] = b.Connection(neuron_groups['Ai'], neuron_groups['Ae'], structure='dense', state = 'g'+'i')
connections[connName].connect(neuron_groups['Ai'], neuron_groups['Ae'], weightMatrix)

rate_monitors['Ae'] = b.PopulationRateMonitor(neuron_groups['Ae'], bin = (single_example_time+resting_time)/b.second)
rate_monitors['Ai'] = b.PopulationRateMonitor(neuron_groups['Ai'], bin = (single_example_time+resting_time)/b.second)
spike_counters['Ae'] = b.SpikeCounter(neuron_groups['Ae'])


#------------------------------------------------------------------------------ 
# create input population and connections from input populations 
#------------------------------------------------------------------------------ 

input_groups['Xe'] = b.PoissonGroup(n_input, 0)
rate_monitors['Xe'] = b.PopulationRateMonitor(input_groups['Xe'], bin = (single_example_time + resting_time) / b.second)

# spike_monitors['Xe'] = b.SpikeMonitor(input_groups['Xe'])

connName = 'XeAe'
weightMatrix = np.load('../random/XeAe.npy')
#connections[connName] = b.Connection(input_groups['Xe'], neuron_groups['Ae'], structure='dense', state = 'g' + 'e', delay=True, max_delay=delay['ee_input'][1])
#connections[connName].connect(input_groups['Xe'], neuron_groups['Ae'], weightMatrix, delay=delay['ee_input'])
connections[connName] = b.Connection(input_groups['Xe'], neuron_groups['Ae'], structure='dense', state = 'ge')
connections[connName].connect(input_groups['Xe'], neuron_groups['Ae'], weightMatrix)

stdp_methods['XeAe'] = b.STDP(connections['XeAe'], eqs=eqs_stdp_ee, pre = eqs_stdp_pre_ee, post = eqs_stdp_post_ee, wmin=0., wmax= wmax_ee)

#------------------------------------------------------------------------------ 
# run the simulation and set inputs
#------------------------------------------------------------------------------ 

previous_spike_count = np.zeros(n_e)
input_groups['Xe'].rate = 0

b.run(0)

j = 0
while j < num_examples:

    normalize_weights()
    rates = training_set[j] * 32. *  input_intensity

    input_groups['Xe'].rate = rates
    b.run(single_example_time)

    current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])

    if np.sum(current_spike_count) < 5:
        input_intensity += 1
    else:
        input_intensity = start_input_intensity
        j += 1

    input_groups['Xe'].rate = 0
    b.run(resting_time)
    
    print ( "----------" )
    print ( j, input_intensity )
    print ( np.sum(np.asarray(spike_counters['Ae'].count[:])) )
    print ( np.asarray(spike_counters['Ae'].count[:]) )
    w = connections[connName][:]
    w = np.copy(w)
    print ( np.std(w), np.max(w), np.min(w) )

# np.save('spks', np.array(spike_monitors['Xe'].spikes))
# print np.count_nonzero(np.array(spike_monitors['Xe'].spikes))

#------------------------------------------------------------------------------ 
# save results
#------------------------------------------------------------------------------ 
print 'save results'
save_theta()
save_connections()




