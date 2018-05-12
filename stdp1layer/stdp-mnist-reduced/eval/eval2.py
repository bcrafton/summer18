'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

import brian as b
from brian import *
import numpy as np
import matplotlib
import matplotlib.cm as cmap
import time
import os.path
import scipy 
import cPickle as pickle
from struct import unpack
import brian.experimental.realtime_monitor as rltmMon
from PIL import Image
from scipy.misc import toimage

'''
arr = np.random.randint(0,256, 100*100)
arr.resize((100,100))
plt.gray()
im =  Image.fromarray(arr, mode="L")
plt.imshow(im)
'''

'''
arr = np.random.randint(0,256, 100*100)
arr.resize((100,100))
im =  Image.fromarray(arr, mode="L")
im.show()
'''

#------------------------------------------------------------------------------ 
# functions
#------------------------------------------------------------------------------     
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

def get_recognized_number_ranking(assignments, spike_rates):

    # print (assignments.shape)
    # print (spike_rates.shape)

    summed_rates = [0] * 10
    num_assignments = [0] * 10

    # for all 10 numbers
    for i in xrange(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            # print(spike_rates[assignments == i].shape)
            # print(num_assignments[i])

            # so we sum the spiking rates of neurons assigned to a number and then normalize it.
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]

            # print(summed_rates[i])

    # sort them least to greatest
    ret = np.argsort(summed_rates)[::-1]
    # print(ret)
    # print('-----------------')

    return ret

def get_new_assignments(result_monitor, input_numbers):
    # 100x400
    # #examples x #pixels or #neurons input layer
    # print result_monitor.shape

    # the answers
    # print( input_numbers )
    # 1x100 array
    # print input_numbers.shape

    assignments = np.ones(n_e) * -1 # initialize them as not assigned

    # input numbers is our input number pics
    input_nums = np.asarray(input_numbers)

    # n_e = 400.
    maximum_rate = [0] * n_e    

    for j in xrange(10):
        # number if input images that are j
        num_inputs = len(np.where(input_nums == j)[0])

        # if > 0 of them
        if num_inputs > 0:
            # all 400 neurons weight for j.
            # print(result_monitor[input_nums == j].shape)
            # print(num_inputs)
            # print( result_monitor[input_nums == j] )

            # print( np.sum(result_monitor[input_nums == j], axis = 0) )

            # summing a (13,400) to get 400, then normalizing it by dividing by number inputs (13)
            # sum ( rates of all neurons when shown all images that are number j ) / num of images that are j
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_inputs
        
        # assign each neuron an image [1-100] it spiked the most at.
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



















