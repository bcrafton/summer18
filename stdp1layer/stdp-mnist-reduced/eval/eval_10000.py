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

def get_rates(result_monitor, input_numbers):
    rates = []
    input_nums = np.asarray(input_numbers)

    for j in xrange(10):
        num_inputs = len(np.where(input_nums == j)[0])

        if num_inputs > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_inputs
            rates.append(rate) 

    return rates

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

MNIST_data_path = '../data/'
data_path = '../activity/'
training_ending = '10000'
testing_ending = '10000'
start_time_training = 0
end_time_training = int(training_ending)
start_time_testing = 0
end_time_testing = int(testing_ending)

n_e = 400
n_input = 784
ending = ''

print 'load MNIST'
training = get_labeled_data(MNIST_data_path + 'training')
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)

print 'load results'
training_result_monitor = np.load(data_path + 'resultPopVecs' + training_ending + ending + '.npy')
training_input_numbers = np.load(data_path + 'inputNumbers' + training_ending + '.npy')
testing_result_monitor = np.load(data_path + 'resultPopVecs' + testing_ending + '.npy')
testing_input_numbers = np.load(data_path + 'inputNumbers' + testing_ending + '.npy')
# print training_result_monitor.shape

print 'get assignments'
test_results = np.zeros((10, end_time_testing-start_time_testing))
test_results_max = np.zeros((10, end_time_testing-start_time_testing))
test_results_top = np.zeros((10, end_time_testing-start_time_testing))
test_results_fixed = np.zeros((10, end_time_testing-start_time_testing))
assignments = get_new_assignments(training_result_monitor[start_time_training:end_time_training], training_input_numbers[start_time_training:end_time_training])

# print (training_input_numbers.shape)

rates = get_rates(training_result_monitor[start_time_training:end_time_training], training_input_numbers[start_time_training:end_time_training])
for i in range(10):
  # print(i)
  arr = np.copy(rates[i])
  arr = arr > 0.01
  arr.resize((20,20))
  im = toimage(arr)
  im.save(str(i) + ".png")


# training gives assignment data
# pretty sure in form of firing rate. 

counter = 0 
num_tests = end_time_testing / 10000
sum_accurracy = [0] * num_tests
while (counter < num_tests):
    end_time = min(end_time_testing, 10000*(counter+1))
    start_time = 10000*counter
    test_results = np.zeros((10, end_time-start_time))
    print 'calculate accuracy for sum'

    for i in xrange(end_time - start_time):
        test_results[:,i] = get_recognized_number_ranking(assignments, testing_result_monitor[i+start_time,:])

    '''
    for i in xrange(end_time - start_time):
        print('--------------')
        print(test_results[:,i])
        print(testing_result_monitor[i+start_time,:])

    for i in range(end_time-start_time):
        print (test_results[0,i])
        print (testing_input_numbers[i])
    '''

    difference = test_results[0,:] - testing_input_numbers[start_time:end_time]
    correct = len(np.where(difference == 0)[0])
    incorrect = np.where(difference != 0)[0]
    sum_accurracy[counter] = correct/float(end_time-start_time) * 100
    print 'Sum response - accuracy: ', sum_accurracy[counter], ' num incorrect: ', len(incorrect)
    counter += 1

print 'Sum response - accuracy --> mean: ', np.mean(sum_accurracy),  '--> standard deviation: ', np.std(sum_accurracy)

'''
arr = np.copy(assignments)
arr = (arr == 1) * 255
arr.resize((20,20))
im = toimage(arr)
im.save("1.png")

arr = np.copy(assignments)
arr = (arr == 2) * 255
arr.resize((20,20))
im = toimage(arr)
im.save("2.png")

arr = np.copy(assignments)
arr = (arr == 3) * 255
arr.resize((20,20))
im = toimage(arr)
im.save("3.png")

arr = np.copy(assignments)
arr = (arr == 4) * 255
arr.resize((20,20))
im = toimage(arr)
im.save("4.png")

arr = np.copy(assignments)
arr = (arr == 5) * 255
arr.resize((20,20))
im = toimage(arr)
im.save("5.png")

arr = np.copy(assignments)
arr = (arr == 6) * 255
arr.resize((20,20))
im = toimage(arr)
im.save("6.png")

arr = np.copy(assignments)
arr = (arr == 7) * 255
arr.resize((20,20))
im = toimage(arr)
im.save("7.png")

arr = np.copy(assignments)
arr = (arr == 8) * 255
arr.resize((20,20))
im = toimage(arr)
im.save("8.png")

arr = np.copy(assignments)
arr = (arr == 9) * 255
arr.resize((20,20))
im = toimage(arr)
im.save("9.png")

#b.show()

'''


















