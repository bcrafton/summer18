
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 

class Dropout(Layer):

    def __init__(self, rate):
        self.rate = rate

    def num_params(self):
        return 0

    def forward(self, X, dropout=False):
        if dropout:
            self.dropout_mask = tf.cast(tf.random_uniform(shape=tf.shape(X)) > self.rate, tf.float32) # np.random.binomial(size=X.shape, n=1, p=1 - self.rate)
            return X * self.dropout_mask
        else:
            return X

    def backward(self, AI, AO, DO):
        return DO * self.dropout_mask
        
    def gv(self, AIN, AOUT, DO):
        return []

    def dfa(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))

    def dfa_gv(self, AI, AO, E, DO):
        return []

