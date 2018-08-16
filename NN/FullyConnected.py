
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class FullyConnected(Layer):
    def __init__(self, size : tuple, weights : np.ndarray, alpha : float, activation : Activation, last_layer : bool):
        
        # TODO
        # check to make sure what we put in here is correct
        
        # input size
        self.size = size
        self.last_layer = last_layer
        self.input_size, self.output_size = size
        
        # weights
        self.weights = weights
        # self.B = B
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

        # print (AIN.get_shape(), AOUT.get_shape(), DIN.get_shape(), DOUT.get_shape(), self.weights.get_shape())

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.matmul(DO, tf.transpose(self.weights))
        DW = tf.matmul(tf.transpose(AI), DO)
        DB = tf.reduce_sum(DO, axis=0)
        self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, DW)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
                
        return DI
        
    def dfa(self, AIN : np.ndarray, AOUT : np.ndarray, DIN : np.ndarray):
        pass
