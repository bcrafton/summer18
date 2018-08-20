
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class FullyConnected(Layer):
    def __init__(self, size : tuple, num_classes : int, weights : np.ndarray, alpha : float, activation : Activation, last_layer : bool, sparse : bool):
        
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
        if sparse:
            b = np.zeros(shape=(self.output_size, self.num_classes))
            for ii in range(self.output_size):
                idx = int(np.random.randint(0, self.num_classes))
                b[ii][idx] = np.random.uniform(-1.0/sqrt_fan_out, 1.0/sqrt_fan_out)
            b = np.transpose(b)
            self.B = tf.cast(tf.Variable(b), tf.float32)        
        else:
            self.B = tf.Variable(tf.random_uniform(shape=[self.num_classes, self.output_size], minval=-1.0/sqrt_fan_out, maxval=1.0/sqrt_fan_out))

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
        return DI
        
    def gv(self, AI : np.ndarray, AO : np.ndarray, DO : np.ndarray):
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        DB = tf.reduce_sum(DO, axis=0)
        return [(DW, self.weights), (DB, self.bias)]
    
    def dfa(self, AI: np.ndarray, AO: np.ndarray, E: np.ndarray, DO: np.ndarray):
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI: np.ndarray, AO: np.ndarray, E: np.ndarray, DO: np.ndarray):
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
        
        
        
        
        
        
        
