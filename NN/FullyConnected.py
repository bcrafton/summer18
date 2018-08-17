
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class FullyConnected(Layer):
    def __init__(self, size : tuple, num_classes : int, weights : np.ndarray, alpha : float, activation : Activation, last_layer : bool):
        
        # TODO
        # check to make sure what we put in here is correct
        
        # input size
        self.size = size
        self.last_layer = last_layer
        self.input_size, self.output_size = size
        self.num_classes = num_classes
        
        # weights
        self.weights = weights
        
        sqrt_fan_out = math.sqrt(self.output_size)
        self.B = tf.Variable(tf.random_uniform(shape=[self.num_classes, self.output_size], minval=-1.0, maxval=1.0))
        
        # self.bias = np.zeros(self.output_size)
        self.bias = tf.Variable(tf.zeros(shape=[self.output_size]))        

        # lr
        self.alpha = alpha
        
        # activation function
        self.activation = activation

    def get_weights(self):
        return [self.weights, self.bias]

    def forward(self, X : np.ndarray, dropout=False):
        Z = tf.matmul(X, self.weights) + self.bias
        A = self.activation.forward(Z)
        return A
            
    def backward(self, AI : np.ndarray, AO : np.ndarray, DO : np.ndarray):

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.matmul(DO, tf.transpose(self.weights))
        DW = tf.matmul(tf.transpose(AI), DO)
        DB = tf.reduce_sum(DO, axis=0)
        self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, DW)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
                
        return DI
        
    def dfa(self, AI : np.ndarray, AO : np.ndarray, DO : np.ndarray):
                
        if self.last_layer:
            DO = tf.multiply(DO, self.activation.gradient(AO))
        else:
            DO = tf.matmul(DO, self.B)
            DO = tf.multiply(DO, self.activation.gradient(AO))
            
        # dropout_mask = tf.cast(tf.random_uniform(shape=tf.shape(DO)) > 0.5, tf.float32)
        # DO = tf.multiply(DO, dropout_mask)
            
        DW = tf.matmul(tf.transpose(AI), DO)
        DB = tf.reduce_sum(DO, axis=0)
        
        self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, DW)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        
        return None
        
        
        
        
        
        
        
        
        
