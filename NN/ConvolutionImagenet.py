
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class Convolution(Layer):
    num = 0
    def __init__(self, input_sizes, filter_sizes, num_classes, init_filters, strides, padding, alpha, activation: Activation, bias, last_layer):
        self.input_sizes = input_sizes
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        
        # self.h and self.w only equal this for input sizes when padding = "SAME"...
        self.batch_size, self.h, self.w, self.fin = self.input_sizes
        self.fh, self.fw, self.fin, self.fout = self.filter_sizes
        
        if init_filters == "zero":
            self.filters = tf.Variable(tf.zeros(shape=self.filter_sizes))
        elif init_filters == "sqrt_fan_in":
            sqrt_fan_in = math.sqrt(self.h*self.w*self.fin)
            self.filters = tf.Variable(tf.random_uniform(shape=self.filter_sizes, minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
        elif init_filters == "epsilon":
            self.filters = tf.Variable(tf.ones(shape=self.filter_sizes) * 1e-9)
        else:
            # self.filters = tf.Variable(tf.random_normal(shape=self.filter_sizes, mean=0.0, stddev=0.01))
            self.filters = tf.get_variable(name="conv" + str(Convolution.num), shape=self.filter_sizes)
            Convolution.num = Convolution.num + 1

        # bias
        self.bias = tf.Variable(tf.ones(shape=self.fout) * bias)

        self.strides = strides
        self.padding = padding
        
        self.alpha = alpha
        
        self.activation = activation
        self.last_layer = last_layer
        
    def num_params(self):
        filter_weights_size = self.fh * self.fw * self.fin * self.fout
        bias_weights_size = self.fout
        return filter_weights_size + bias_weights_size
        
    def forward(self, X, dropout=False):
        Z = tf.add(tf.nn.conv2d(X, self.filters, self.strides, self.padding), tf.reshape(self.bias, [1, 1, self.fout]))
        A = self.activation.forward(Z)
        return A
        
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_sizes, filter=self.filters, out_backprop=DO, strides=self.strides, padding=self.padding)
        return DI

    def gv(self, AI, AO, DO):    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])
        return [(DF, self.filters), (DB, self.bias)]
        
    def train(self, AI, AO, DO): 
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        # DF = tf.Print(DF, [tf.reduce_mean(DF), tf.keras.backend.std(DF), tf.reduce_mean(self.filters), tf.keras.backend.std(self.filters)], message="Conv: ")
        # DF = tf.Print(DF, [tf.shape(DF), tf.shape(self.filters)], message="", summarize=25)

        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DF, self.filters), (DB, self.bias)]
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI, AO, E, DO):
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])
        return [(DF, self.filters), (DB, self.bias)]
        
    def dfa(self, AI, AO, E, DO): 
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        # DF = tf.Print(DF, [tf.reduce_mean(DF), tf.keras.backend.std(DF), tf.reduce_mean(self.filters), tf.keras.backend.std(self.filters)], message="Conv: ")
        # DF = tf.Print(DF, [tf.shape(DF), tf.shape(self.filters)], message="", summarize=25)

        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DF, self.filters), (DB, self.bias)]
        
    ###################################################################    
        
        
        
        
