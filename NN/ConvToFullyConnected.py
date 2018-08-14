
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 

class ConvToFullyConnected(Layer):

    def __init__(self, shape):
        self.shape = shape

    def forward(self, X: np.ndarray):
        return tf.reshape(X, [X.shape[0], -1])

    def backward(self, AIN : np.ndarray, AOUT : np.ndarray, DO : np.ndarray):
        return tf.reshape(DO, [32, 28, 28, 16])

    def dfa(self, AIN : np.ndarray, AOUT : np.ndarray, DO : np.ndarray):
        pass


