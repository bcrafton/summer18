
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

single_example_time =   0.35 * b.second
resting_time = 0.15 * b.second

v_rest_e = -65. * b.mV 
v_rest_i = -60. * b.mV 
v_reset_e = -65. * b.mV
v_reset_i = -45. * b.mV
v_thresh_e = -52. * b.mV
v_thresh_i = -40. * b.mV
refrac_e = 5. * b.ms
refrac_i = 2. * b.ms

conn_structure = 'dense'

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

class SNN:

  def __init__(self):

    self.neuron_groups = {}
    self.input_groups = {}
    self.connections = {}
    self.stdp_methods = {}
    # self.rate_monitors = {}
    self.spike_counters = {}

    np.random.seed(0)

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

    self.neuron_groups['e'] = b.NeuronGroup(400 * NUM_LAYERS, neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= scr_e, compile = True, freeze = True)
    self.neuron_groups['i'] = b.NeuronGroup(400 * NUM_LAYERS, neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i, compile = True, freeze = True)

    #------------------------------------------------------------------------------ 
    # create network population and recurrent connections
    #------------------------------------------------------------------------------ 

    self.neuron_groups['Ae'] = self.neuron_groups['e'].subgroup(400)
    self.neuron_groups['Ai'] = self.neuron_groups['i'].subgroup(400)

    self.neuron_groups['Ae'].v = v_rest_e - 40. * b.mV
    self.neuron_groups['Ai'].v = v_rest_i - 40. * b.mV
    self.neuron_groups['e'].theta = np.ones((400)) * 20.0*b.mV

    connName = 'AeAi'
    weightMatrix = np.load('../random/AeAi.npy')
    self.connections[connName] = b.Connection(self.neuron_groups['Ae'], self.neuron_groups['Ai'], structure= conn_structure, state = 'g'+'e')
    self.connections[connName].connect(self.neuron_groups['Ae'], self.neuron_groups['Ai'], weightMatrix)

    connName = 'AiAe'
    weightMatrix = np.load('../random/AiAe.npy')
    self.connections[connName] = b.Connection(self.neuron_groups['Ai'], self.neuron_groups['Ae'], structure= conn_structure, state = 'g'+'i')
    self.connections[connName].connect(self.neuron_groups['Ai'], self.neuron_groups['Ae'], weightMatrix)

    # self.rate_monitors['Ae'] = b.PopulationRateMonitor(self.neuron_groups['Ae'], bin = (single_example_time+resting_time)/b.second)
    # self.rate_monitors['Ai'] = b.PopulationRateMonitor(self.neuron_groups['Ai'], bin = (single_example_time+resting_time)/b.second)
    self.spike_counters['Ae'] = b.SpikeCounter(self.neuron_groups['Ae'])


    #------------------------------------------------------------------------------ 
    # create input population and connections from input populations 
    #------------------------------------------------------------------------------ 

    self.input_groups['Xe'] = b.PoissonGroup(28*28, 0)
    #self.rate_monitors['Xe'] = b.PopulationRateMonitor(self.input_groups['Xe'], bin = (single_example_time + resting_time) / b.second)

    connName = 'XeAe'
    weightMatrix = np.load('../random/XeAe.npy')
    self.connections[connName] = b.Connection(self.input_groups['Xe'], self.neuron_groups['Ae'], structure=conn_structure, state = 'g' + 'e', delay=True, max_delay=delay['ee_input'][1])
    self.connections[connName].connect(self.input_groups['Xe'], self.neuron_groups['Ae'], weightMatrix, delay=delay['ee_input'])

    self.stdp_methods['XeAe'] = b.STDP(self.connections['XeAe'], eqs=eqs_stdp_ee, pre = eqs_stdp_pre_ee, post = eqs_stdp_post_ee, wmin=0., wmax= wmax_ee)

    #------------------------------------------------------------------------------ 
    #------------------------------------------------------------------------------ 

    print "Inst SNN"

  def train(rates):
    normalize_weights()

    current_spike_count = np.zeros(400)
    input_intensity = start_input_intensity

    while np.sum(current_spike_count) < 5:
      self.input_groups['Xe'].rate = rates
      b.run(single_example_time, report='text')

      current_spike_count = np.asarray(self.spike_counters['Ae'].count[:]) - previous_spike_count
      previous_spike_count = np.copy(self.spike_counters['Ae'].count[:])

      input_intensity += 1

    self.input_groups['Xe'].rate = 0
    b.run(resting_time)

  def save():
    connMatrix = self.connections['XeAe'][:]
    np.save('XeAe', connMatrix)
    np.save('theta_A', self.neuron_groups['Ae'].theta)

snn = SNN()

