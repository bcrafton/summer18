
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class Convolution(Layer):
    def __init__(self, input_sizes, filter_sizes, num_classes, filters, stride, padding, alpha, activation: Activation=None, last_layer=False):
        self.input_sizes = input_sizes
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        
        # self.h and self.w only equal this for input sizes when padding = "SAME"...
        self.batch_size, self.h, self.w, self.fin = self.input_sizes
        self.fh, self.fw, self.fin, self.fout = self.filter_sizes
        
        self.filters = filters
        sqrt_fan_out = math.sqrt(self.fout * self.h * self.w)
        self.B = tf.Variable(tf.random_uniform(shape=[self.num_classes, self.h*self.w*self.fout], minval=-1.0/sqrt_fan_out, maxval=1.0/sqrt_fan_out))
        
        # TODO
        self.stride = stride
        self.stride = [1,1,1,1]
        
        # TODO
        self.padding = padding
        self.padding = "SAME"
        
        self.alpha = alpha
        
        self.activation = activation
        self.last_layer = last_layer
        
    def get_weights(self):
        return self.filters
        
    def forward(self, X : np.ndarray, dropout=False):
        Z = tf.nn.conv2d(X, self.filters, self.stride, self.padding)
        A = self.activation.forward(Z)
        # A = tf.Print(A, [A], message="this is a: ")
        return A
        
    def backward(self, AI: np.ndarray, AO: np.ndarray, DO: np.ndarray):    
        # apply activation gradient
        DO = tf.multiply(DO, self.activation.gradient(AO))
        # DO = tf.Print(DO, [tf.metrics.mean(DO)], message=": ")
        
        # send this back
        # DI = tf.nn.conv2d_backprop_input(input_sizes=[BATCH_SIZE, 28, 28, 1], filter=W, out_backprop=Y, strides=[1,1,1,1], padding="SAME")
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_sizes, filter=self.filters, out_backprop=DO, strides=self.stride, padding="SAME")
        
        # update with this
        # DF = tf.nn.conv2d_backprop_filter(input=X, filter_sizes=[5, 5, 1, 16], out_backprop=Y, strides=[1,1,1,1], padding="SAME")        
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.stride, padding="SAME")
        
        # update filters
        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        
        # return error wtr to input 
        return DI

    def dfa(self, AI: np.ndarray, AO: np.ndarray, DO: np.ndarray):

        DO = tf.matmul(DO, self.B)
        # DO = tf.reshape(DO, [self.batch_size, self.f, self.h, self.w])
        DO = tf.reshape(DO, [self.batch_size, self.h, self.w, self.fout])
        # DO = tf.Print(DO, [tf.metrics.mean(DO)], message="DO: ")
        DO = tf.multiply(DO, self.activation.gradient(AO))
        
        dropout_mask = tf.cast(tf.random_uniform(shape=tf.shape(DO)) > 0.5, tf.float32)
        DO = DO * dropout_mask
        
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.stride, padding="SAME")
        # DF = tf.Print(DF, [tf.metrics.mean(DF)], message="DF: ")
        
        # update filters
        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        
        return None
        
        
        
        
        
        
        
        
        
        
