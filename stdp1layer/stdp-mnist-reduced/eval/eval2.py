'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

import numpy as np
import matplotlib
import matplotlib.cm as cmap
import time
import os.path
import scipy 
import cPickle as pickle
from struct import unpack

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

start_time_training = 0
end_time_training = 1000

start_time_testing = 0
end_time_testing = 1000

n_e = 400
n_input = 784
ending = ''

print 'load results'
training_result_monitor = np.load('../activity/resultPopVecs1000.npy')
training_input_numbers = np.load('../activity/inputNumbers1000.npy')

testing_result_monitor = np.load('../activity/resultPopVecs1000.npy')
testing_input_numbers = np.load('../activity/inputNumbers1000.npy')

print 'get assignments'
test_results = np.zeros((10, 1000))
test_results_max = np.zeros((10, 1000))
test_results_top = np.zeros((10, 1000))
test_results_fixed = np.zeros((10, 1000))
assignments = get_new_assignments(training_result_monitor[0:1000], training_input_numbers[0:1000])
print assignments
print testing_input_numbers

end_time = 1000
start_time = 0

test_results = np.zeros((10, 1000))

for i in xrange(1000):
    test_results[:, i] = get_recognized_number_ranking(assignments, testing_result_monitor[i, :])

difference = test_results[0, :] - testing_input_numbers[0:1000]
correct = len(np.where(difference == 0)[0])
incorrect = np.where(difference != 0)[0]

accuracy = correct/float(end_time-start_time) * 100

print accuracy



















