
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
        self.bias = np.zeros(self.output_size)
        
        # lr
        self.alpha = alpha
        
        # activation function
        self.activation = activation

    def forward(self, batch_size : int, X : np.ndarray):
        Z = tf.matmul(X, self.weights) + self.bias
        A = self.activation.forward(Z)
        return A
            
    def backward(self, batch_size : int, AIN : np.ndarray, AOUT : np.ndarray, DIN : np.ndarray):

        # print (AIN.get_shape(), AOUT.get_shape(), DIN.get_shape(), DOUT.get_shape(), self.weights.get_shape())

        DIN = tf.multiply(DIN, self.activation.gradient(AOUT))
        DOUT = tf.matmul(DIN, tf.transpose(self.weights))
        G = tf.matmul(tf.transpose(AIN), DIN)
        self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, G)))
        # WE NEED TO UPDATE OUR BIAS...
        
        return DOUT
        
    def dfa(self, AIN : np.ndarray, AOUT : np.ndarray, DIN : np.ndarray):
        pass
