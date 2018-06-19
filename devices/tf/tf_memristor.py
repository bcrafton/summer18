import os
import sys
import time

import tensorflow as tf

import numpy as np
import pylab as plt

##############################################

U = 1e-16
D = 10e-9
W0 = 5e-9
RON = 5e4
ROFF = 1e6
P = 5

steps = 1500
T = 2 * np.pi
dt = T / steps

W = tf.ones(shape=(100, 100)) * W0

Ts = tf.linspace(0.0, T, steps)
Ts = tf.reshape(Ts, [1500, 1, 1])
Vs = tf.ones(shape=(1500, 100, 100)) * tf.sin(Ts) 

# Is = tf.Variable( tf.zeros(shape=(1500, 100, 100)) )
# Rs = tf.Variable( tf.zeros(shape=(1500, 100, 100)) )

Is = tf.zeros(shape=(1500, 100, 100))
Rs = tf.zeros(shape=(1500, 100, 100))

##############################################

def condition(t, W, Vs, Is, Rs):
    return t < 1500

def body(t, W, Vs, Is, Rs):
    V = Vs[t]
    R = RON * (W / D) + ROFF * (1 - (W / D))
    I = V / R
    
    F = 1 - (2 * (W / D) - 1) ** (2 * P)
    dwdt = ((U * RON * I) / D) * F
    W += dwdt * dt

    # Is += tf.one_hot(t, 1) * I
    # Rs += tf.one_hot(t, 1) * R

    return t+1, W, Vs, Is, Rs
    
grad = tf.while_loop(condition, body, [tf.constant(0), W, Vs, Is, Rs])

##############################################

config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

##############################################

start = time.time()
t, W, Vs, Is, Rs = sess.run(grad)
end = time.time()

# print (W)
print end - start

##############################################

# Vplot = Vs[:, 0, 0].eval(session=sess)
# Iplot = Is[:, 0, 0].eval(session=sess)

# Vplot = Vs[:, 0, 0]
# Iplot = Is[:, 0, 0]

# plt.plot(Vplot, Iplot)
# plt.show()







