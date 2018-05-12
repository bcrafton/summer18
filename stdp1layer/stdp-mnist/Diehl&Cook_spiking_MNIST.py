'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

 
import numpy as np
import matplotlib.cm as cmap
import time
import os.path
import scipy 
import cPickle as pickle
import brian_no_units  #import it to deactivate unit checking --> This should NOT be done for testing/debugging 
import brian as b
from struct import unpack
import gzip
from brian import *

# specify the location of the MNIST data
MNIST_data_path = ''

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
   
def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]
    
        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in xrange(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in xrange(cols)]  for unused_row in xrange(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]
            
        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data

def get_matrix_from_file(fileName):
    offset = len(ending) + 4
    if fileName[-4-offset] == 'X':
        n_src = n_input                
    else:
        if fileName[-3-offset]=='e':
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-1-offset]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(fileName)
    print readout.shape, fileName
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr


def save_connections(ending = ''):
    print 'save connections'
    for connName in save_conns:
        connMatrix = connections[connName][:]
#         connListSparse = ([(i,j[0],j[1]) for i in xrange(connMatrix.shape[0]) for j in zip(connMatrix.rowj[i],connMatrix.rowdata[i])])
        connListSparse = ([(i,j,connMatrix[i,j]) for i in xrange(connMatrix.shape[0]) for j in xrange(connMatrix.shape[1]) ])
        np.save(data_path + 'weights/' + connName + ending, connListSparse)

def save_theta(ending = ''):
    print 'save theta'
    for pop_name in population_names:
        np.save(data_path + 'weights/theta_' + pop_name + ending, neuron_groups[pop_name + 'e'].theta)

def normalize_weights():
    for connName in connections:
        if connName[1] == 'e' and connName[3] == 'e':
            connection = connections[connName][:]
            temp_conn = np.copy(connection)
            colSums = np.sum(temp_conn, axis = 0)
            colFactors = weight['ee_input']/colSums
            for j in xrange(n_e):#
                connection[:,j] *= colFactors[j]
            
def get_2d_input_weights():
    name = 'XeAe'
    weight_matrix = np.zeros((n_input, n_e))
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    connMatrix = connections[name][:]
    weight_matrix = np.copy(connMatrix)
        
    for i in xrange(n_e_sqrt):
        for j in xrange(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights


def plot_2d_input_weights():
    name = 'XeAe'
    weights = get_2d_input_weights()
    fig = b.figure(fig_num, figsize = (18, 18))
    im2 = b.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    b.colorbar(im2)
    b.title('weights of connection' + name)
    fig.canvas.draw()
    return im2, fig
    
def update_2d_input_weights(im, fig):
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im

def get_current_performance(performance, current_example_num):
    current_evaluation = int(current_example_num/update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance

def plot_performance(fig_num):
    num_evaluations = int(num_examples/update_interval)
    time_steps = range(0, num_evaluations)
    performance = np.zeros(num_evaluations)
    fig = b.figure(fig_num, figsize = (5, 5))
    fig_num += 1
    ax = fig.add_subplot(111)
    im2, = ax.plot(time_steps, performance) #my_cmap
    b.ylim(ymax = 100)
    b.title('Classification performance')
    fig.canvas.draw()
    return im2, performance, fig_num, fig

def update_performance_plot(im, performance, current_example_num, fig):
    performance = get_current_performance(performance, current_example_num)
    im.set_ydata(performance)
    fig.canvas.draw()
    return im, performance
    
def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in xrange(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    
    for j in xrange(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
        for i in xrange(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments
    
    
#------------------------------------------------------------------------------ 
# load MNIST
#------------------------------------------------------------------------------
start = time.time()
training = get_labeled_data(MNIST_data_path + 'training')
end = time.time()
print 'time needed to load training set:', end - start
 
start = time.time()
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)
end = time.time()
print 'time needed to load test set:', end - start

load_data()

#------------------------------------------------------------------------------ 
# set parameters and equations
#------------------------------------------------------------------------------
test_mode = True

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
data_path = './'
if test_mode:
    weight_path = data_path + 'weights/'
    num_examples = 1000 * 1
    use_testing_set = True
    do_plot_performance = False
    record_spikes = True
    ee_STDP_on = False
    update_interval = num_examples
else:
    weight_path = data_path + 'random/'  
    num_examples = 60000 * 3
    use_testing_set = False
    do_plot_performance = True
    if num_examples <= 60000:    
        record_spikes = True
    else:
        record_spikes = True
    ee_STDP_on = True


ending = ''
n_input = 784
n_e = 400
n_i = n_e 
single_example_time =   0.35 * b.second #
resting_time = 0.15 * b.second
runtime = num_examples * (single_example_time + resting_time)
if num_examples <= 10000:    
    update_interval = num_examples
    weight_update_interval = 20
else:
    update_interval = 10000
    weight_update_interval = 100
if num_examples <= 60000:    
    save_connections_interval = 10000
else:
    save_connections_interval = 10000
    update_interval = 10000

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
delay = {}
input_population_names = ['X']
population_names = ['A']
input_connection_names = ['XA']
save_conns = ['XeAe']
input_conn_names = ['ee_input'] 
recurrent_conn_names = ['ei', 'ie']
weight['ee_input'] = 78.
delay['ee_input'] = (0*b.ms,10*b.ms)
delay['ei_input'] = (0*b.ms,5*b.ms)
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

if test_mode:
    scr_e = 'v = v_reset_e; timer = 0*ms'
else:
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
if test_mode:
    neuron_eqs_e += '\n  theta      :volt'
else:
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
result_monitor = np.zeros((update_interval,n_e))

neuron_groups['e'] = b.NeuronGroup(n_e*len(population_names), neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= scr_e, 
                                   compile = True, freeze = True)
neuron_groups['i'] = b.NeuronGroup(n_i*len(population_names), neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i, 
                                   compile = True, freeze = True)


#------------------------------------------------------------------------------ 
# create network population and recurrent connections
#------------------------------------------------------------------------------ 

neuron_groups['Ae'] = neuron_groups['e'].subgroup(n_e)
neuron_groups['Ai'] = neuron_groups['i'].subgroup(n_i)

neuron_groups['Ae'].v = v_rest_e - 40. * b.mV
neuron_groups['Ai'].v = v_rest_i - 40. * b.mV
neuron_groups['e'].theta = np.load('./weights/theta_A.npy')

# recurrent connections
'''
connName = 'AeAi'
tmp = np.load('../random/AeAi.npy')
avg = np.average(tmp)
std = np.std(tmp)
weightMatrix = np.random.normal(avg, std, size=(n_e, n_i))
# print weightMatrix

# this does not work.
# weightMatrix = np.load('../random/AeAi.npy')

connections[connName] = b.Connection(neuron_groups['Ae'], neuron_groups['Ai'], structure= conn_structure, state = 'g'+'e')
connections[connName].connect(neuron_groups['Ae'], neuron_groups['Ai'], weightMatrix)
'''

connName = 'AeAi'
weightMatrix = get_matrix_from_file('./random/AeAi.npy')
connections[connName] = b.Connection(neuron_groups['Ae'], neuron_groups['Ai'], structure= conn_structure, state = 'g'+'e')
connections[connName].connect(neuron_groups['Ae'], neuron_groups['Ai'], weightMatrix)

'''
connName = 'AiAe'
tmp = np.load('../random/AiAe.npy')
avg = np.average(tmp)
std = np.std(tmp)
weightMatrix = np.random.normal(avg, std, size=(n_i, n_e))

# this does not work 
# weightMatrix = np.load('../random/AiAe.npy')
connections[connName] = b.Connection(neuron_groups['Ai'], neuron_groups['Ae'], structure= conn_structure, state = 'g'+'i')
connections[connName].connect(neuron_groups['Ai'], neuron_groups['Ae'], weightMatrix)
'''

connName = 'AiAe'
weightMatrix = get_matrix_from_file('./random/AiAe.npy')
connections[connName] = b.Connection(neuron_groups['Ai'], neuron_groups['Ae'], structure= conn_structure, state = 'g'+'i')
connections[connName].connect(neuron_groups['Ai'], neuron_groups['Ae'], weightMatrix)

rate_monitors['Ae'] = b.PopulationRateMonitor(neuron_groups['Ae'], bin = (single_example_time+resting_time)/b.second)
rate_monitors['Ai'] = b.PopulationRateMonitor(neuron_groups['Ai'], bin = (single_example_time+resting_time)/b.second)
spike_counters['Ae'] = b.SpikeCounter(neuron_groups['Ae'])

spike_monitors['Ae'] = b.SpikeMonitor(neuron_groups['Ae'])
spike_monitors['Ai'] = b.SpikeMonitor(neuron_groups['Ai'])

b.figure(fig_num)
fig_num += 1
b.ion()
b.subplot(211)
b.raster_plot(spike_monitors['Ae'], refresh=1000*b.ms, showlast=1000*b.ms)
b.subplot(212)
b.raster_plot(spike_monitors['Ai'], refresh=1000*b.ms, showlast=1000*b.ms)

#------------------------------------------------------------------------------ 
# create input population and connections from input populations 
#------------------------------------------------------------------------------ 

input_groups['Xe'] = b.PoissonGroup(n_input, 0)
rate_monitors['Xe'] = b.PopulationRateMonitor(input_groups['Xe'], bin = (single_example_time + resting_time) / b.second)

connName = 'XeAe'
weightMatrix = get_matrix_from_file('./weights/XeAe.npy')
connections[connName] = b.Connection(input_groups['Xe'], neuron_groups['Ae'], structure=conn_structure, state = 'g' + 'e', delay=True, max_delay=delay['ee_input'][1])
connections[connName].connect(input_groups['Xe'], neuron_groups['Ae'], weightMatrix, delay=delay['ee_input'])

'''
#------------------------------------------------------------------------------ 
# create input population and connections from input populations 
#------------------------------------------------------------------------------ 
pop_values = [0,0,0]
for i,name in enumerate(input_population_names):
    input_groups[name+'e'] = b.PoissonGroup(n_input, 0)
    rate_monitors[name+'e'] = b.PopulationRateMonitor(input_groups[name+'e'], bin = (single_example_time+resting_time)/b.second)

for name in input_connection_names:
    print 'create connections between', name[0], 'and', name[1]
    for connType in input_conn_names:
        connName = name[0] + connType[0] + name[1] + connType[1]
        print connName
        weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')
        connections[connName] = b.Connection(input_groups[connName[0:2]], neuron_groups[connName[2:4]], structure= conn_structure, 
                                                    state = 'g'+connType[0], delay=True, max_delay=delay[connType][1])
        connections[connName].connect(input_groups[connName[0:2]], neuron_groups[connName[2:4]], weightMatrix, delay=delay[connType])
     
    if ee_STDP_on:
        print 'create STDP for connection', name[0]+'e'+name[1]+'e'
        stdp_methods[name[0]+'e'+name[1]+'e'] = b.STDP(connections[name[0]+'e'+name[1]+'e'], eqs=eqs_stdp_ee, pre = eqs_stdp_pre_ee, 
                                                       post = eqs_stdp_post_ee, wmin=0., wmax= wmax_ee)
'''

#------------------------------------------------------------------------------ 
# run the simulation and set inputs
#------------------------------------------------------------------------------ 

previous_spike_count = np.zeros(n_e)
assignments = np.zeros(n_e)
input_numbers = [0] * num_examples

# outputNumbers = np.zeros((num_examples, 10))

input_groups['Xe'].rate = 0
b.run(0)
j = 0

while j < num_examples:

    print j

    rates = testing_set[j] * 32. *  input_intensity
    print np.average(rates)

    input_groups['Xe'].rate = rates
    b.run(single_example_time, report='text')
            
    if j % update_interval == 0 and j > 0:
        assignments = get_new_assignments(result_monitor[:], input_numbers[j - update_interval : j])

    current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])

    if np.sum(current_spike_count) < 5:
        input_intensity += 1

    else:
        result_monitor[j, :] = current_spike_count
        input_numbers[j] = testing_labels[j]

        '''
        outputNumbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j % update_interval, :])
        '''

        input_intensity = start_input_intensity
        j += 1

    input_groups['Xe'].rate = 0
    b.run(resting_time)

#------------------------------------------------------------------------------ 
# save results
#------------------------------------------------------------------------------ 
print 'save results'
if not test_mode:
    save_theta()
if not test_mode:
    save_connections()
else:
    np.save(data_path + 'activity/resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + 'activity/inputNumbers' + str(num_examples), input_numbers)
    

#------------------------------------------------------------------------------ 
# plot results
#------------------------------------------------------------------------------ 
'''
if rate_monitors:
    b.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(rate_monitors):
        b.subplot(len(rate_monitors), 1, i)
        b.plot(rate_monitors[name].times/b.second, rate_monitors[name].rate, '.')
        b.title('Rates of population ' + name)
    
if spike_monitors:
    b.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_monitors):
        b.subplot(len(spike_monitors), 1, i)
        b.raster_plot(spike_monitors[name])
        b.title('Spikes of population ' + name)
        
if spike_counters:
    b.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_counters):
        b.subplot(len(spike_counters), 1, i)
        b.plot(spike_counters['Ae'].count[:])
        b.title('Spike count of population ' + name)

plot_2d_input_weights()
b.ioff()
b.show()
'''



