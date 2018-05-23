import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Model parameters
dt = 0.5
d = 8.0   
a = 0.02
c = -65.0
b = 0.2
T = int(1000 // dt)
v_init = -65
u_init = -14.0

v_shape = [1]
u_shape = v_shape
I_app_shape = v_shape

## Make graph
with tf.Graph().as_default() as tf_graph:
    
    ## Variables in model
    v = tf.Variable(tf.ones(shape=v_shape) * v_init, dtype=tf.float32, name='v')
    u = tf.Variable(tf.ones(shape=v_shape) * u_init, dtype=tf.float32, name='u')
    fired = tf.Variable(np.zeros(v_shape, dtype=bool), dtype=tf.bool, name='fired')
    
    ## Inputs to the model
    I_app = tf.placeholder(tf.float32, shape=I_app_shape)
    
    ## Computation
    # Reset any that spiked last timestep
    v_in = tf.where(fired, tf.ones(tf.shape(v))*c, v)
    u_in = tf.where(fired, tf.ones(tf.shape(u))*tf.add(u, d), u)
    
    dv = tf.subtract(tf.add(tf.multiply(tf.add(tf.multiply(0.04, v_in), 5.0), v_in), 140), u_in)
    v_updated = tf.add(v_in, tf.multiply(tf.add(dv, I_app), dt))
    
    du = tf.multiply(a, tf.subtract(tf.multiply(b, v_in), u_in))
    u_out = tf.add(u_in, tf.multiply(dt, du))
    
    # Deal with spikes
    # Limit anything above threshold to threshold value (35)
    # We are saving which fired to use again in the next iteration
    fired_op = fired.assign(tf.greater_equal(v_updated, tf.ones(tf.shape(v)) * 35))
    v_out = tf.where(fired_op, tf.ones(tf.shape(v)) * 35, v_updated)
    
    v_op = v.assign(v_out)
    u_op = u.assign(u_out)
    
## Session 

vs = [np.array(v_init).reshape(1)]
us = [np.array(u_init).reshape(1)]
fires = [np.array(u_init).reshape(1)]

with tf.Session(graph=tf_graph) as sess:
    sess.run(tf.global_variables_initializer())
    
    for t in range(T):
        
        if t * dt > 200 and t * dt < 700:
            iapp = 7.0
        else:
            iapp = 0.0
            
        feed = {I_app: np.array(iapp).reshape(1)}
        
        vo, uo, fire = sess.run([v_op, u_op, fired_op], feed_dict=feed)
        vs.append(vo)
        us.append(uo)
        fires.append(fire)
        
plt.plot([np.asscalar(x) for x in vs])
plt.show()



