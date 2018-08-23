
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class Feedback(Layer):
    def __init__(self, size, num_classes, sparse):
        self.size = size
        self.num_classes = num_classes
        self.sparse = sparse
        self.batch_size, self.h, self.w, self.f = self.size

        sqrt_fan_out = math.sqrt(self.f * self.h * self.w)
        if self.sparse:
            b = np.zeros(shape=(self.f * self.h * self.w, self.num_classes))
            for ii in range(self.f * self.h * self.w):
                idx = int(np.random.randint(0, self.num_classes))
                b[ii][idx] = np.random.uniform(-1.0/sqrt_fan_out, 1.0/sqrt_fan_out)
            b = np.transpose(b)
            self.B = tf.cast(tf.Variable(b), tf.float32)
        else:
            self.B = tf.Variable(tf.random_uniform(shape=[self.num_classes, self.f * self.h * self.w], minval=-1.0/sqrt_fan_out, maxval=1.0/sqrt_fan_out))

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
        E = tf.reshape(E, self.size)
        E = tf.multiply(E, DO)
        return E
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
        
        
        
        
        
        
        
