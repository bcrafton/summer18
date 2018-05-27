
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import cPickle as pickle
import gzip

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
dt = 10
time_steps = int(T / dt)

#############

load_data()
Wsyn = np.random.normal(0, 1.0, size=(28*28, 200))
Wsyn = np.absolute(Wsyn)
Wsyn = Wsyn * (1e-2 / np.average(Wsyn))

#############

ex_number = 0
start_input_intensity = 5
input_intensity = start_input_intensity

while ex_number < 5000:

    prev = Wsyn

    ################
    Isyn = np.zeros(28*28)
    g = np.zeros(shape=(1, 28*28))
    v = np.zeros(200)
    u = np.zeros(200)

    input_fires = np.zeros(shape=(time_steps, 28*28))
    input_fired_counts = np.zeros(28*28)

    output_fired = np.zeros(200)
    output_not_fired = np.zeros(200)
    output_fires = np.zeros(shape=(time_steps, 200))
    output_fired_counts = np.zeros(200)

    # what is freq / ms
    # Hz / 1000
    # (25.0 / 1000.0) = 25Hz
    # rates = state * (25.0 / 1000.0)

    rates = (training_set[ex_number] * 32.)
    rates = rates / (np.max(rates)) # divide by max rate
    rates = rates * input_intensity
    rates = rates / 1000 # divide to get ms

    for t in range(time_steps):
        input_fired = np.random.rand(1, 28*28) < rates * dt
        input_fires[t] = input_fired
        input_fired_counts = input_fired_counts + input_fired

        Isyn = np.dot(input_fired, Wsyn)

        dv = 0.001 * v * v
        dv = 0
        v = np.clip(v + (dv + Isyn) * dt, 0, 35)

        # print (np.max(Isyn), np.average(Isyn))
        # print (np.max(v), np.average(v))

        output_fired = v >= 35
        output_not_fired = v < 35
        output_fires[t] = output_fired
        output_fired_counts = output_fired_counts + output_fired

        v = v * output_not_fired
        v = v + output_fired * 0

    output_fires_post = np.zeros(shape=(time_steps,200))
    output_fires_pre = np.zeros(shape=(time_steps,200))

    for i in range(200):
        flag = 0
        for j in range(time_steps):
            if (output_fires[j][i]):
                flag = 8
            if flag:
                output_fires_post[j][i] = 1
                flag = flag - 1

        flag = 0
        for j in reversed(range(time_steps)):
            if (output_fires[j][i]):
                flag = 8
            if flag:
                output_fires_pre[j][i] = 1
                flag = flag - 1

    
    input_fires = np.transpose(input_fires)
    output_fires_post = np.transpose(output_fires_post)
    output_fires_pre = np.transpose(output_fires_pre)

    post = np.zeros(shape=(28*28,200))
    pre = np.zeros(shape=(28*28,200))

    for i in range(200):
        for j in range(28*28):
            post[j][i] = np.count_nonzero(np.logical_and(input_fires[j], output_fires_post[i]))
            pre[j][i] = np.count_nonzero(np.logical_and(input_fires[j], output_fires_pre[i]))

    gradient = (pre - post) * (1e-3)
    Wsyn = Wsyn + gradient
    Wsyn = Wsyn * (1e-2 / np.average(Wsyn))

    if np.sum(output_fired_counts) < 10:
        input_intensity += 1
        print (np.average(input_fired_counts), np.max(input_fired_counts), np.sum(input_fired_counts))
        print (np.average(output_fired_counts), np.max(output_fired_counts), np.sum(output_fired_counts))
    else:
        #################
        print ("---------")
        print (ex_number, input_intensity)

        print (np.average(pre), np.max(pre), np.sum(pre))
        print (np.average(post), np.max(post), np.sum(post))
        print (np.average(pre - post), np.max(pre - post), np.sum(pre - post))

        print (np.average(input_fired_counts), np.max(input_fired_counts), np.sum(input_fired_counts))
        print (np.average(output_fired_counts), np.max(output_fired_counts), np.sum(output_fired_counts))

        print ( np.average(np.absolute(Wsyn)),     np.max(np.absolute(Wsyn))     )
        print ( np.average(np.absolute(gradient)), np.max(np.absolute(gradient)) )

        # print (np.average(Wsyn))
        # print (np.sum(Wsyn))
        # print (np.average(Wsyn - prev))
        # print (np.sum(Wsyn - prev))
        #################
        input_intensity = start_input_intensity
        ex_number += 1








