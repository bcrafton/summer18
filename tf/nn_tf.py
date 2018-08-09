
import tensorflow as tf
import numpy as np
import math
  
def add_bias(x):
  return tf.concat([x, tf.ones([tf.shape(x)[0], 1])], axis=1)
  
def minus_bias(x):
  return x[:, :-1]
  
class nn_tf:
    def __init__(self, size, weights, alpha, bias):
        # check to make sure we have the right number of layers and weights
        self.num_layers = len(size)
        self.num_weights = len(weights)
        assert(self.num_layers-1 == self.num_weights)
        
        # check to make sure that the sizes of the layers matches up
        for ii in range(1, self.num_layers):
            if bias:
                shape1 = (size[ii-1]+1, size[ii])
                shape2 = tf.shape(weights[ii-1])
            else:
                shape1 = (size[ii-1], size[ii])
                shape2 = tf.shape(weights[ii-1])
            # assert(shape1 == shape2)
        
        self.size = size
        self.weights = weights
        self.alpha = alpha
        self.bias = bias
        
    def predict(self, x):
        A = [None] * self.num_layers
        Z = [None] * self.num_layers
    
        for ii in range(self.num_layers):
            if ii == 0:
                Z[ii] = None
                A[ii] = add_bias(x) if self.bias else x
            elif ii == self.num_layers-1:
                Z[ii] = tf.matmul(A[ii-1], self.weights[ii-1])
                A[ii] = tf.sigmoid(Z[ii])
            else:
                Z[ii] = tf.matmul(A[ii-1], self.weights[ii-1])
                A[ii] = add_bias(tf.sigmoid(Z[ii])) if self.bias else tf.sigmoid(Z[ii])
                
        return A[self.num_layers-1]

    
    def train(self, x, y):
        A = [None] * self.num_layers
        Z = [None] * self.num_layers
        D = [None] * self.num_layers
        G = [None] * self.num_layers
    
        for ii in range(self.num_layers):
            if ii == 0:
                Z[ii] = None
                A[ii] = add_bias(x) if self.bias else x
            elif ii == self.num_layers-1:
                Z[ii] = tf.matmul(A[ii-1], self.weights[ii-1])
                A[ii] = tf.sigmoid(Z[ii])
            else:
                Z[ii] = tf.matmul(A[ii-1], self.weights[ii-1])
                A[ii] = add_bias(tf.sigmoid(Z[ii])) if self.bias else tf.sigmoid(Z[ii])

        for ii in range(self.num_layers-1, 0, -1):

            if ii == self.num_layers-1:
                D[ii] = tf.subtract(A[ii], y)
                G[ii-1] = tf.matmul(tf.transpose(A[ii-1]), D[ii])
            else:
                D[ii] = tf.matmul(D[ii+1], tf.transpose(self.weights[ii]))
                if self.bias:
                    D[ii] = minus_bias(D[ii])
                D[ii] = tf.multiply(D[ii], tf.multiply(tf.sigmoid(Z[ii]), tf.subtract(1.0, tf.sigmoid(Z[ii]))))

            G[ii-1] = tf.matmul(tf.transpose(A[ii-1]), D[ii])
            self.weights[ii-1] = self.weights[ii-1].assign(tf.subtract(self.weights[ii-1], tf.scalar_mul(1e-4, G[ii-1])))
        
        return self.weights
            
                
                
                

        
