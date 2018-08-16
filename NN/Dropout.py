
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 

class Dropout(Layer):

    def __init__(self, rate):
        self.rate = rate

    def initialize(self, input_size, out_layer_size, train_method):
        return input_size

    def forward(self, X):
        if mode == 'train':
            self.dropout_mask = np.random.binomial(size=X.shape, n=1, p=1 - self.rate)
            return X * self.dropout_mask
        else:
            return X

    def dfa(self, E: np.ndarray):
        return None

    def back_prob(self, E: np.ndarray):
        return E * self.dropout_mask, 0, 0



