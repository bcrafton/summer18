
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 

class ConvToFullyConnected(Layer):

    def __init__(self, shape):
        self.shape = shape
        
    def get_weights(self):
        return tf.random_uniform(shape=(1, 1))

    def forward(self, X, dropout=False):
        return tf.reshape(X, [tf.shape(X)[0], -1])

    def backward(self, AIN, AOUT, DO):
        return tf.reshape(DO, [tf.shape(AIN)[0]] + self.shape)

    def gv(self, AIN, AOUT, DO):
        return []

    def dfa(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))

    def dfa_gv(self, AI, AO, E, DO):
        return []
