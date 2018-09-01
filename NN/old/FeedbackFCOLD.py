
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class FeedbackFC(Layer):
    def __init__(self, size : tuple, num_classes : int, sparse : bool, rank : int):
        self.size = size
        self.num_classes = num_classes
        self.sparse = sparse
        self.rank = rank
        self.input_size, self.output_size = self.size

        sqrt_fan_out = math.sqrt(self.output_size)
        if self.rank > 0:
            b = np.zeros(shape=(self.output_size, self.num_classes))
            for ii in range(self.rank):
                tmp1 = np.random.uniform(-np.sqrt(sqrt_fan_out), np.sqrt(sqrt_fan_out), size=(self.output_size, 1))
                tmp2 = np.random.uniform(-np.sqrt(sqrt_fan_out), np.sqrt(sqrt_fan_out), size=(1, self.num_classes))
                b = b + (1.0 / self.rank) * np.dot(tmp1, tmp2)
            b = np.transpose(b)
            self.B = tf.cast(tf.Variable(b), tf.float32)
            
        elif sparse:
            b = np.zeros(shape=(self.output_size, self.num_classes))
            for ii in range(self.output_size):
                idx = int(np.random.randint(0, self.num_classes))
                b[ii][idx] = np.random.uniform(-1.0/sqrt_fan_out, 1.0/sqrt_fan_out)
            b = np.transpose(b)
            self.B = tf.cast(tf.Variable(b), tf.float32)
              
        else:
            self.B = tf.Variable(tf.random_uniform(shape=[self.num_classes, self.output_size], minval=-1.0/sqrt_fan_out, maxval=1.0/sqrt_fan_out))

    def num_params(self):
        return 0
        
    def forward(self, X, dropout=False):
        return X
        
    def backward(self, AI, AO, DO):    
        return DO

    def gv(self, AI, AO, DO):    
        return []

    def dfa(self, AI, AO, E, DO):
        E = tf.matmul(E, self.B)
        E = tf.multiply(E, DO)
        return E
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
        
        
        
        
        
        
        

