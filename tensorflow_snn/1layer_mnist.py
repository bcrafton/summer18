
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
Wsyn = np.random.normal(128.0, 10.0, size=(28*28, 20*20)) / 128 / 100

#############

ex_number = 0
input_intensity = 2

while ex_number < 5000:

    prev = Wsyn

    # if accuracy above N then break

    ################
    Isyn = np.zeros(28*28)
    g = np.zeros(shape=(1, 28*28))
    v = np.zeros(20*20)
    u = np.zeros(20*20)

    input_fires = np.zeros(shape=(time_steps, 28*28))
    input_fired_counts = np.zeros(28*28)

    output_fired = np.zeros(20*20)
    output_not_fired = np.zeros(20*20)
    output_fires = np.zeros(shape=(time_steps,20*20))
    output_fired_counts = np.zeros(20*20)

    # what is freq / ms
    # Hz / 1000
    # (25.0 / 1000.0) = 25Hz
    # rates = state * (25.0 / 1000.0)

    rates = (training_set[ex_number] * 32. *  input_intensity)
    rates = rates / 32 # average was 128, so 128/rate... 128/2 = 64
    rates = rates / 1000 # divide to get ms
    # rates = rates / (28*28 / 16)

    # print(np.max(rates))
    # print(np.average(rates))

    for t in range(time_steps):
        input_fired = np.random.rand(1, 28*28) < rates * dt
        input_fires[t] = input_fired
        input_fired_counts = input_fired_counts + input_fired

        # g = g + input_fired
        Isyn = np.dot(input_fired, Wsyn)
        Isyn = Isyn - (np.dot(input_fired, Wsyn) * v)
        Isyn = Isyn.flatten()
        # g = (1 - dt/10) * g

        print (np.max(Isyn))
        # print (np.max(Wsyn))
        # print ("voltage", str(np.max(v)))

        dv = np.clip((0.04 * v + 5) * v + 140 - u, -10e12, 10e12)
        v = np.clip(v + (dv + Isyn) * dt, 36, -65)
        du = 0.02 * (0.2 * v - u)
        u = u + dt * du

        output_fired = v > 35
        output_not_fired = v < 35
        output_fires[t] = output_fired
        output_fired_counts = output_fired_counts + output_fired

        v = v * output_not_fired
        v = v + output_fired * -65
        u = u + output_fired * 8

    if np.average(output_fired_counts) < 3:
        input_intensity *= 2
    else:
        input_intensity = 2
        ex_number += 1

    output_fires_post = np.zeros(shape=(time_steps,20*20))
    output_fires_pre = np.zeros(shape=(time_steps,20*20))

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
    output_fires_post = np.transpose(output_fires_post)
    output_fires_pre = np.transpose(output_fires_pre)

    post = np.zeros(shape=(28*28,20*20))
    pre = np.zeros(shape=(28*28,20*20))

    for i in range(20*20):
        for j in range(28*28):
            post[j][i] = np.count_nonzero(np.logical_and(input_fires[j], output_fires_post[i]))
            pre[j][i] = np.count_nonzero(np.logical_and(input_fires[j], output_fires_pre[i]))

    gradient = pre - post
    Wsyn = (9 * Wsyn + gradient) / 9

    #################

    print ("---------")
    print (ex_number)

    print (np.average(pre), np.max(pre))
    print (np.average(post), np.max(post))

    print (np.average(input_fired_counts), np.max(input_fired_counts))
    print (np.average(output_fired_counts), np.max(output_fired_counts))

    print (np.average(Wsyn))
    print (np.sum(Wsyn))
    print (np.average(Wsyn - prev))
    print (np.sum(Wsyn - prev))
        
    #################













