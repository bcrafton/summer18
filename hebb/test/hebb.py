
import numpy as np
import math
import numpy as np
import cPickle as pickle
import gzip
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--iters', type=int, default=1)
parser.add_argument('--examples', type=int, default=10000)
parser.add_argument('--hi', type=float, default=10.0)
parser.add_argument('--lo', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.0001)
args = parser.parse_args()

# iters = (args.iters if (args.iters is not None) else 1)
# after default:
# iters = args.iters

layer1 = 28*28
layer2 = 20*20

print ("---------")
print (args.iters)
print (args.examples)
print (args.hi)
print (args.lo)
print (args.lr)

#####################################

def load_data():
  global training_set, training_labels, testing_set, testing_labels
  f = gzip.open('mnist.pkl.gz', 'rb')
  train, valid, test = pickle.load(f)

  [training_set, training_labels] = train
  [validation_set, validation_labels] = valid
  [testing_set, testing_labels] = test

  for i in range( len(training_set) ):
    training_set[i] = training_set[i].reshape(layer1)

  for i in range( len(testing_set) ):
    testing_set[i] = testing_set[i].reshape(layer1)

  f.close()

load_data()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#####################################

w = np.absolute(np.random.normal(1.0, 0.25, size=(layer1, layer2)))

#####################################
prev = np.copy(w)
for itr in range(args.iters):
  for i in range(args.examples):
    x = np.array(training_set[i]).reshape(layer1, 1)
    x = x / np.average(x)

    xw = np.dot(np.transpose(x), w)
    xw = xw / np.max(xw)
    xw = np.power(xw, 3)
    
    sig = sigmoid(xw)
    e = sig - 0.5 * xw
    w += args.lr * np.dot(x, e)
    w = np.clip(w, args.lo, args.hi)

    col_norm = np.average(w, axis = 0)
    col_norm = 1.0 / col_norm
    for j in range(layer2):
      w[:, j] *= col_norm[j]
      
    if (i % 999 == 0):
        # print (np.sum(np.absolute(w - prev)), np.sum(w), np.sum(np.absolute(w - prev)) / np.sum(w))
        prev = np.copy(w)
#####################################
max_rates = np.zeros(layer2)
assignments = np.zeros(layer2)

for i in range(args.examples):
  x = np.array(training_set[i]).reshape(layer1, 1)
  x = x / np.average(x)
  
  xw = np.dot(np.transpose(x), w)
  xw = xw / np.max(xw)
  xw = np.power(xw, 3)
  xw = np.array(xw).flatten()
  
  for j in range(layer2):
    if max_rates[j] < xw[j]:
      max_rates[j] = xw[j]
      assignments[j] = training_labels[i]
      
# print (assignments)
#####################################
correct = 0

for i in range(args.examples):
  x = np.array(training_set[i]).reshape(layer1, 1)
  x = x / np.average(x)
  
  xw = np.dot(np.transpose(x), w)
  xw = xw / np.max(xw)
  xw = np.power(xw, 3)

  max_spikes = 0
  for j in range(10):
    mask = assignments == j
    num_assignments = np.count_nonzero(mask)
    spikes = xw * mask
    
    if np.sum(spikes):
      spikes = np.sum(spikes) / num_assignments
    else:
      spikes = 0
      
    if max_spikes < spikes:
      max_spikes = spikes
      predict = j
  
  correct += predict == training_labels[i]

print (np.std(w))
print (1.0 * correct) / args.examples
#####################################

