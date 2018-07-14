

import numpy as np
import matplotlib
import matplotlib.cm as cmap
import time
import os.path
import scipy 
import cPickle as pickle
from struct import unpack
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_spks', type=str)
parser.add_argument('--train_labels', type=str)
parser.add_argument('--test_spks', type=str)
parser.add_argument('--test_labels', type=str)
args = parser.parse_args()

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

for num in (100, 1000, 9990):
    # print 'load results'
    training_result_monitor = np.load(args.train_spks)[0:num, :]
    training_input_numbers = np.load(args.train_labels)[0:num]
    testing_result_monitor = np.load(args.test_spks)[0:num, :]
    testing_input_numbers = np.load(args.test_labels)[0:num]
    
    num_examples = len(training_result_monitor)

    # print 'get assignments'
    assignments = get_new_assignments(training_result_monitor[0:num_examples], training_input_numbers[0:num_examples])
    # print assignments

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


















