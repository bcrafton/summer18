
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 

class ConvToFullyConnected(Layer):

    def __init__(self, shape):
        self.shape = shape

    def forward(self, batch_size : int, X: np.ndarray):
        return tf.reshape(X, [batch_size, -1])

    def backward(self, batch_size : int, AIN : np.ndarray, AOUT : np.ndarray, DO : np.ndarray):
        return tf.reshape(DO, [batch_size] + self.shape)

    def dfa(self, AIN : np.ndarray, AOUT : np.ndarray, DO : np.ndarray):
        pass


