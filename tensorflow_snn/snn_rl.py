
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dt = 0.5
d = 8.0   
a = 0.02
c = -65.0
b = 0.2
T = int(1000 // dt)

v_init = np.ones([4, 1]) * -70.0 
u_init = np.ones([4, 1]) * -14.0 

vs = [v_init]
us = [u_init]
fires = []

Wsyn = np.ones(shape=(16, 4)) 

with tf.Graph().as_default() as tf_graph:
    
    ## Variables in model
    v = tf.Variable(tf.ones(shape=(4)) * -70.0 , dtype=tf.float32, name='v')
    u = tf.Variable(tf.ones(shape=(4)) * -14.0 , dtype=tf.float32, name='u')
    fired = tf.Variable(np.zeros(4, dtype=bool), dtype=tf.bool, name='fired')
    
    ## Inputs to the model
    I_app = tf.placeholder(tf.float32, shape=(4))
    
    ## Computation
    # fired ? -65   : v.
    v_in = tf.where(fired, tf.ones(tf.shape(v)) * -65.0, v)
    # fired ? u + 8 : u.
    u_in = tf.where(fired, tf.ones(tf.shape(u)) * tf.add(u, 8.0), u)
    
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


with tf.Session(graph=tf_graph) as sess:
    sess.run(tf.global_variables_initializer())

    episode = np.ones(16) * 0.5
    for e in range(1):
        rates = episode[e]

        for t in range(T):
            f = np.random.rand(1, 16) < rates * dt
            
            # dont think we need this because we will have decay function
            # Isyn = Isyn + (f * Wsyn) - (0.25 * Wsyn)
            Isyn = np.dot(f, Wsyn)
            feed = {I_app: np.array(Isyn).reshape(4)}

            vo, uo, fire = sess.run([v_op, u_op, fired_op], feed_dict=feed)
            vs.append(vo)
            us.append(uo)

            for ii in range(len(fire)):
                if fire[ii]:
                    fires.append([ii, t * dt])
                
           
print (fires)




        
