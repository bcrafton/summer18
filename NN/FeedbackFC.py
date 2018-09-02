
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

        if self.rank and self.sparse:
            assert(self.rank >= self.sparse)

        #### CREATE THE SPARSE MASK ####
        if self.sparse:
            self.mask = np.zeros(shape=(self.output_size, self.num_classes))
            for ii in range(self.output_size):
                if self.rank > 0:
                    idx = np.random.randint(0, self.rank, size=self.sparse)
                else:
                    idx = np.random.randint(0, self.num_classes, size=self.sparse)
                self.mask[ii][idx] = 1.0
                
            self.mask = np.transpose(self.mask)
        else:
            self.mask = np.ones(shape=(self.num_classes, self.output_size))
        
        #### IF MATRIX HAS USER-SPECIFIED RANK ####
        sqrt_fan_out = np.sqrt(self.output_size)
        
        if self.rank > 0:
            lo = -1.0/np.sqrt(sqrt_fan_out)
            hi = 1.0/np.sqrt(sqrt_fan_out)
            
            b = np.zeros(shape=(self.output_size, self.num_classes))
            for ii in range(self.rank):
                tmp1 = np.random.uniform(lo, hi, size=(self.output_size, 1))
                tmp2 = np.random.uniform(lo, hi, size=(1, self.num_classes))
                b = b + (1.0 / self.rank) * np.dot(tmp1, tmp2)
                
            b = np.transpose(b)
            b = b * self.mask
            assert(np.linalg.matrix_rank(b) == self.rank)
            
            self.B = tf.cast(tf.Variable(b), tf.float32)
        else:
            lo = -1.0/sqrt_fan_out
            hi = 1.0/sqrt_fan_out
        
            b = np.random.uniform(lo, hi, size=(self.num_classes, self.output_size))
            b = b * self.mask
            self.B = tf.cast(tf.Variable(b), tf.float32)

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
        
        
        
        
        
        
        
