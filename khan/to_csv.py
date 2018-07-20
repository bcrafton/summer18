
import numpy as np
import cPickle as pickle
import gzip

def load_data():
  global training_set, training_labels, testing_set, testing_labels
  f = gzip.open('mnist.pkl.gz', 'rb')
  train, valid, test = pickle.load(f)
  f.close()

  [training_set, training_labels] = train
  [validation_set, validation_labels] = valid
  [testing_set, testing_labels] = test

  for i in range( len(training_set) ):
    training_set[i] = training_set[i].reshape(28*28)

  for i in range( len(testing_set) ):
    testing_set[i] = testing_set[i].reshape(28*28)
    
##################

load_data()

##################

np.savetxt("training_set.csv",    training_set,    delimiter=" ")
np.savetxt("training_labels.csv", training_labels, delimiter=" ")
np.savetxt("testing_set.csv",     testing_set,     delimiter=" ")
np.savetxt("testing_labels.csv",  testing_labels,  delimiter=" ")
