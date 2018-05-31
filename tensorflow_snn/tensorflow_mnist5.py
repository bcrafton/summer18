
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import cPickle as pickle
import gzip
import threading
from collections import deque

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


T = 1000
dt = 2
time_steps = int(T / dt)

# max_weight = 5e-4
# min_weight = 5e-6

max_weight = 8e-4
min_weight = 4e-6

avg_weight = 5e-5
learning_rate = 5e-9

#############

load_data()
Wsyn = np.random.normal(0, 1.0, size=(28*28, 20*20))
Wsyn = np.absolute(Wsyn)
# Wsyn = np.load('XeAe.npy')
assert( (np.prod(Wsyn.shape) - np.count_nonzero(Wsyn >= 0)) == 0 )
Wsyn = Wsyn * (avg_weight / np.average(Wsyn))

Wsyn_e = np.random.normal(0, 1.0, size=(20*20, 20*20))
Wsyn_e = np.absolute(Wsyn_e)
Wsyn_e = Wsyn_e * (avg_weight / np.average(Wsyn_e))

Wsyn_i = np.random.normal(0, 1.0, size=(20*20, 20*20))
Wsyn_i = np.absolute(Wsyn_i)
Wsyn_i = Wsyn_i * (avg_weight / np.average(Wsyn_i))

#############

def calc_gradient(idx, start_idx, end_idx, gradient, spikes, labels):
    start_input_intensity = 10
    input_intensity = start_input_intensity
    
    gradient[idx] = np.zeros(shape=(28*28, 20*20))
    spikes[idx] = []
    labels[idx] = []

    while start_idx < end_idx:
    
        print ("thread #" + str(idx) + " " + str(start_idx) + " " + str(input_intensity))

        ################
        v = np.zeros(20*20)
        u = np.zeros(20*20)
        
        v_i = np.zeros(20*20)
        u_i = np.zeros(20*20)
        
        Isyn = np.zeros(28*28)
        g = np.zeros(shape=(1, 28*28))
        
        Isyn_e = np.zeros(20*20)
        g_e = np.zeros(20*20)
        
        Isyn_i = np.zeros(20*20)
        g_i = np.zeros(shape=(1, 20*20))

        ################
        
        input_fires = np.zeros(shape=(time_steps, 28*28))
        input_fired_counts = np.zeros(28*28)
        
        ################
        
        output_fired = np.zeros(20*20)
        output_not_fired = np.zeros(20*20)
        output_fires = np.zeros(shape=(time_steps, 20*20))
        output_fired_counts = np.zeros(20*20)
        refractory = np.zeros(20*20)
        
        ################
        
        inh_fired = np.zeros(20*20)
        inh_not_fired = np.zeros(20*20)
        inh_fires = np.zeros(shape=(time_steps, 20*20))
        inh_fired_counts = np.zeros(20*20)
        inh_refractory = np.zeros(20*20)
        
        ################

        # what is freq / ms
        # Hz / 1000
        # (25.0 / 1000.0) = 25Hz
        # rates = state * (25.0 / 1000.0)

        rates = (training_set[start_idx] * 32.)
        rates = rates / (np.average(rates)) # divide by max rate
        rates = rates * input_intensity
        rates = rates / 1000 # divide to get ms

        for t in range(time_steps):
        
            ####################################
        
            input_fired = np.random.rand(1, 28*28) < rates * dt
            input_fires[t] = input_fired
            input_fired_counts = input_fired_counts + input_fired
            
            ####################################

            g = g + input_fired
            Isyn = np.dot(g, Wsyn)
            Isyn = Isyn.flatten ()
            g = np.max((1 - dt/10) * g, 0)
            
            g_e = g_e + inh_fired
            Isyn_e = np.dot(g_e, Wsyn_e) # this is the 400x400 synapses
            Isyn_e = Isyn_e.flatten()
            g_e = np.max((1 - dt/10) * g_e, 0)

            dv = -0.07 * dt
            v = np.clip(v + (dv + Isyn - Isyn_e) * dt, 0, 35)
            # v = np.clip(v + (dv + Isyn) * dt, 0, 35)
            not_refractory = (refractory == 0)
            v = v * not_refractory

            output_fired = v >= 35
            output_not_fired = v < 35
            output_fires[t] = output_fired
            output_fired_counts = output_fired_counts + output_fired
            
            refractory = refractory + output_fired * 10
            refractory = np.clip(refractory - 1, 0, 4)

            v = v * output_not_fired

            ####################################
            
            g_i = g_i + output_fired
            Isyn_i = np.dot(g_i, Wsyn_i)
            Isyn_i = Isyn_i.flatten()
            g_i = np.max((1 - dt/5) * g_i, 0)
            
            # print ("exc", np.average(g), np.average(Isyn))
            # print ("inh", np.average(g_i), np.average(Isyn_i))

            dv_i = -0.07 * dt
            v_i = np.clip(v_i + (dv_i + Isyn_i) * dt, 0, 35)
            not_inh_refractory = (inh_refractory == 0)
            v_i = v_i * not_inh_refractory

            inh_fired = v_i >= 35
            inh_not_fired = v_i < 35
            inh_fires[t] = inh_fired
            inh_fired_counts = inh_fired_counts + inh_fired
            
            inh_refractory = inh_refractory + inh_fired * 10
            inh_refractory = np.clip(inh_refractory - 1, 0, 4)

            v_i = v_i * inh_not_fired
            
            ####################################

        output_fires_post = np.zeros(shape=(time_steps, 20*20))
        output_fires_pre = np.zeros(shape=(time_steps, 20*20))

        for i in range(20*20):
            flag = 0
            for j in range(time_steps):
                if (output_fires[j][i]):
                    flag = 10
                if flag:
                    output_fires_post[j][i] = 1
                    flag = flag - 1

            flag = 0
            for j in reversed(range(time_steps)):
                if (output_fires[j][i]):
                    flag = 10
                if flag:
                    output_fires_pre[j][i] = 1
                    flag = flag - 1

        
        input_fires = np.transpose(input_fires)
        post = np.dot(input_fires, output_fires_post)
        pre = np.dot(input_fires, output_fires_pre)
        
        ##############################################

        if np.sum(output_fired_counts) < 10:
            input_intensity += 1
        else:
            print ( np.sum(input_fired_counts), np.sum(output_fired_counts), np.sum(inh_fired_counts) )
            print ( np.average(input_fired_counts), np.average(output_fired_counts), np.average(inh_fired_counts) )
        
            input_intensity = start_input_intensity
            
            gradient[idx] = gradient[idx] + (pre - post) * learning_rate
            spikes[idx].append(output_fired_counts.flatten())
            labels[idx].append(training_labels[start_idx])
            
            start_idx = start_idx + 1
    
#############

assignments = np.zeros(20*20)
all_spikes = deque(maxlen=1200)
all_labels = deque(maxlen=1200)
ex_number = 0
n_threads = 4

while ex_number < 24000:
    threads = []
    
    gradient = [None] * n_threads
    spikes = [None] * n_threads
    labels = [None] * n_threads

    prev = Wsyn
    
    for t in range(n_threads):
        start_idx = ex_number + t * 20
        end_idx = ex_number + (t + 1) * 20
        thread = threading.Thread(target=calc_gradient, args=(t, start_idx, end_idx, gradient, spikes, labels))
        threads.append(thread)
        
        print(t, start_idx, end_idx)
        
    for t in range(n_threads):
        threads[t].start()
    
    for t in range(n_threads):
        threads[t].join()
    
    print ("averages: ")
    print (np.average(np.absolute(gradient[0])))
    print (np.average(np.absolute(gradient[1])))
    print (np.average(np.absolute(gradient[2])))
    print (np.average(np.absolute(gradient[3])))
    print (np.average(np.absolute(Wsyn)))
    print ("min and max: ")
    print (np.max(Wsyn), np.min(Wsyn))

    for t in range(n_threads):
        Wsyn = Wsyn + gradient[t]
        all_spikes.extend(spikes[t])
        all_labels.extend(labels[t])

    Wsyn = np.clip(Wsyn, min_weight, max_weight)

    col_norm = np.average(Wsyn, axis = 0)
    col_norm = avg_weight / col_norm
    for i in range(20*20):
        Wsyn[:, i] *= col_norm[i]

    print ("difference: ")
    print (np.sum(np.absolute(Wsyn - prev)), np.sum(np.absolute(Wsyn)))
    # assert( np.sum(np.absolute(Wsyn - prev)) > (np.sum(np.absolute(Wsyn)) * 0.01) )

    ex_number += n_threads * 20

    ##############################################

    if ( ex_number >= 1 and (ex_number % 2400 == 1200) ):
        assignments = np.zeros(20*20)
        maximum_rate = np.zeros(20*20)

        for num in range(10):
            idx = np.where(np.asarray(all_labels) == num)[0]
            num_assignments = len(idx)

            if num_assignments > 0:
                rate = (1.0 * np.sum(np.asarray(all_spikes)[idx], axis = 0)) / num_assignments

                for out_neuron_idx in range(20*20):
                    if maximum_rate[out_neuron_idx] < rate[out_neuron_idx]:
                        maximum_rate[out_neuron_idx] = rate[out_neuron_idx]
                        assignments[out_neuron_idx] = num

        print (maximum_rate)
        print (assignments)

    if ( ex_number >= 1 and (ex_number % 2400 == 0) ):
        correct = 0
        for ex in range(1200):
            spike_sums = np.zeros(10)
            for num in range(10):
                idx = np.where(np.asarray(assignments) == num)[0]
                if ( len(idx) > 0 ):
                    spike_sums[num] = 1.0 * np.sum(all_spikes[ex][idx]) / len(idx)

            predict = np.argsort(spike_sums)
            correct += (predict[-1] == all_labels[ex])

            print (predict, predict[-1], all_labels[ex])

        print (1.0 * correct / 1200)

        #################








