
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 

class ConvToFullyConnected(Layer):

    def __init__(self, shape):
        self.shape = shape
        
    def get_weights(self):
        return tf.random_uniform(shape=(1, 1))

    def forward(self, X: np.ndarray):
        return tf.reshape(X, [tf.shape(X)[0], -1])

    def backward(self, AIN : np.ndarray, AOUT : np.ndarray, DO : np.ndarray):
        return tf.reshape(DO, [tf.shape(AIN)[0]] + self.shape)

    def dfa(self, AIN : np.ndarray, AOUT : np.ndarray, DO : np.ndarray):
        pass


