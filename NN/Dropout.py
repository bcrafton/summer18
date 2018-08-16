
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 

class Dropout(Layer):

    def __init__(self, rate):
        self.rate = rate

    def get_weights(self):
        return tf.random_uniform(shape=(1, 1))

    def forward(self, X : np.ndarray, dropout=False):
        if dropout:
            self.dropout_mask = np.random.binomial(size=X.shape, n=1, p=1 - self.rate)
            return X * self.dropout_mask
        else:
            return X

    def backward(self, AI : np.ndarray, AO : np.ndarray, DO : np.ndarray):
        return DO * self.dropout_mask

    def dfa(self, AI : np.ndarray, AO : np.ndarray, DO : np.ndarray):
        return None

