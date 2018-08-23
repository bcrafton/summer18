
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class BatchNorm(Layer):
    def __init__(self, size : tuple, num_classes : int, alpha : float):
        self.size = size
        self.input_size, self.output_size = size
        self.num_classes = num_classes

        self.bias = tf.Variable(tf.zeros(shape=self.size))        
        self.scale = tf.Variable(tf.zeros(shape=self.size))  
        self.alpha = alpha

    def get_weights(self):
        return []

    def forward(self, X, dropout=False):
        mean, var = tf.nn.moments(x, axes=[1])
        norm = tf.multiply(tf.subtract(X, mean), tf.sqrt(tf.add(var, epsilon)))
        return tf.add(tf.multiply(scale, norm), bias)
            
    def backward(self, AI, AO, DO):
        mean, var = tf.nn.moments(x, axes=[1])
        norm = tf.multiply(tf.subtract(X, mean), tf.sqrt(tf.add(var, epsilon)))
        dbias = tf.reduce_sum(DO, axis=0)
        dscale = np.sum(tf.multiply(norm * dy,) axis=0)
        dh = (1. / N) * gamma * (var + eps)**(-1. / 2.) * (N * dy - np.sum(dy, axis=0) - (h - mu) * (var + eps)**(-1.0) * np.sum(dy * (h - mu), axis=0))
        
    def gv(self, AI, AO, DO):
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        DB = tf.reduce_sum(DO, axis=0)
        return [(DW, self.weights), (DB, self.bias)]
    
    def dfa(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI, AO, E, DO):
        if self.last_layer:
            E = tf.multiply(E, self.activation.gradient(AO))
        else:
            E = tf.matmul(E, self.B)
            E = tf.multiply(E, self.activation.gradient(AO))
            
        # dropout_mask = tf.cast(tf.random_uniform(shape=tf.shape(E)) > 0.5, tf.float32)
        # E = tf.multiply(E, dropout_mask)
            
        DW = tf.matmul(tf.transpose(AI), E)
        DB = tf.reduce_sum(E, axis=0)
        
        return [(DW, self.weights), (DB, self.bias)]
        
        
        
        
        
        
        
