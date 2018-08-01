

import numpy as np
import matplotlib
import matplotlib.cm as cmap
import time
import os.path
import scipy 
from struct import unpack
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--examples', type=int)
#parser.add_argument('--assign_spks', type=str, default='./results/assign_spks_1000.npy')
#parser.add_argument('--assign_labels', type=str, default='./results/assign_labels_1000.npy')
#parser.add_argument('--test_spks', type=str, default='./results/test_spks_1000.npy')
#parser.add_argument('--test_labels', type=str, default='./results/test_labels_1000.npy')
args = parser.parse_args()

assert(args.examples is not None)
assign_spks =   './results/assign_spks_' + str(args.examples) + '.npy'
assign_labels = './results/assign_labels_' + str(args.examples) + '.npy'
test_spks =     './results/test_spks_' + str(args.examples) + '.npy'
test_labels =   './results/test_labels_' + str(args.examples) + '.npy'

#------------------------------------------------------------------------------ 
# functions
#------------------------------------------------------------------------------  

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in xrange(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    ret = np.argsort(summed_rates)[::-1]
    return ret

def get_new_assignments(result_monitor, input_numbers):
    assignments = np.ones(n_e) * -1 # initialize them as not assigned
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    
    for j in xrange(10):
        num_inputs = len(np.where(input_nums == j)[0])
        if num_inputs > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_inputs
        for i in xrange(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j 
    return assignments

n_e = 400
n_input = 784
ending = ''

training_result_monitor = np.load(assign_spks)[:, :]
training_input_numbers = np.load(assign_labels)[:]
testing_result_monitor = np.load(test_spks)[:, :]
testing_input_numbers = np.load(test_labels)[:]

num_examples = len(training_result_monitor)
assignments = get_new_assignments(training_result_monitor[:], training_input_numbers[:])

test_results = np.zeros((10, num_examples))
for i in xrange(num_examples):
    test_results[:, i] = get_recognized_number_ranking(assignments, testing_result_monitor[i, :])

difference = test_results[0, :] - testing_input_numbers[0:num_examples]
correct = len(np.where(difference == 0)[0])
incorrect = np.where(difference != 0)[0]

accuracy = correct/float(num_examples) * 100
total_spks = np.sum(testing_result_monitor)

print "accuracy: " + str(accuracy)
print "total spks: " + str(total_spks)
print "spikes per: " + str(total_spks / num_examples)


















