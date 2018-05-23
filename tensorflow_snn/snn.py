
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

dt = 0.5
n = 1000
inh = np.random.rand(n, 1) < 0.2  # Percent inhibitory
exc = np.logical_not(inh)
d = 8.0 * exc + 2.0 * inh
a = 0.02 * exc + 0.1 * inh
c = -65.0
T = int(1000 // dt)
v_init = np.ones([n, 1]) * -70.0 
u_init = np.ones([n, 1]) * -14.0 
tp = tf.float32

n_in = 100
rate = 2*1e-3 
tau_g = 10.0

# g_in = np.zeros([n_in, 1])
# g_in = np.ones([n_in, 1])
g_in = np.random.rand(n_in, 1)

E_in = np.zeros([n_in, 1])
w_in = 0.07 * np.ones([n, n_in])
w_in[np.random.rand(n, n_in)>0.1] = 0  # input conection prob

# New to step 4
# g_cond = np.zeros([n, 1])
g_cond = np.random.rand(n, 1)

E = np.zeros([n, 1])
E[inh] = -85.0
idx = np.random.rand(n, n) < 0.05  # Percent recurrent connections
W = idx * np.random.gamma(2, 0.003, size=[n, n])
inh2exc = np.matmul(inh, np.transpose(exc))
W += (W * inh2exc * 2)

with tf.Graph().as_default() as g:
    p04 = tf.constant(0.04, dtype=tp)
    five = tf.constant(5.0, dtype=tp)
    one_fort = tf.constant(140.0, dtype=tp)
    b = tf.constant(0.2, dtype=tp)
    dt_g = tf.constant(dt, dtype=tp)
    a_g = tf.constant(dt, dtype=tp)


    v_in = tf.placeholder(tp, shape=[n, 1])
    u_prev = tf.placeholder(tp, shape=[n, 1])
    I_app = tf.placeholder(tp, shape=[n, 1])  
    
    # Reset any above threshold (35)
    units_fired = tf.greater_equal(v_in, tf.ones(tf.shape(v_in)) * 35)
    v_prev = tf.where(units_fired, tf.ones(tf.shape(v_in))*c, v_in)

    dv = tf.subtract(tf.add(tf.multiply(tf.add(tf.multiply(p04, v_prev), five), 
                                        v_prev), one_fort), u_prev)
    v1 = tf.add(dv, I_app)
    v2 = tf.multiply(v1, dt_g)
    v = tf.add(v_prev, v2)

    du1 = tf.multiply(b, v_prev)
    du2 = tf.subtract(du1, u_prev)
    du = tf.multiply(a_g, du2)
    u = tf.add(u_prev, tf.multiply(dt_g, du))
    

    ## if spike
    threshold = tf.constant(35, dtype=tp)
    cond = tf.greater_equal(v, tf.ones(tf.shape(v)) * 35)
    v_out = tf.where(cond, tf.ones(tf.shape(v)) * 35, v)
    u_out = tf.where(cond, tf.ones(tf.shape(u))*tf.add(u_prev, d), u)



vs = [v_init]  # now v_init is shape=(n,1)
us = [u_init]
fired = np.zeros([n, 1])
var_names = ['p', 'fired', 'iapp', 'Isyn', 'g_in', 'g_cond', 'vs[-1]', 'we_calc', "wg_calc"]
stats = {name:[] for name in var_names }
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    
    for t in range(T):    ## Timesteps
        
        if t * dt > 200 and t*dt < 700:
            p = np.random.rand(n_in, 1) < rate*dt
        else:
            p = np.zeros([n_in,1]) 
            
        # Set up which fire
        # g_in = g_in + p
        iapp = np.matmul(w_in, np.multiply(g_in + p, E_in))
        iapp = iapp - np.multiply(np.matmul(w_in, g_in + p), vs[-1])
        # g_in = (1 - dt / tau_g) * g_in
        
        # g_cond = g_cond + fired
        e_calc = np.multiply(g_cond + fired, E)
        we_calc = np.matmul(W, e_calc)
        wg_calc = np.matmul(W, g_cond + fired)   
        Isyn = we_calc - np.multiply(wg_calc, vs[-1])
        iapp += Isyn
        # g_cond = (1 - dt / tau_g) * g_cond
        
        for name in var_names:
            exec("stats['{0}'].append({0}.mean())".format(name))
        
        feed = {v_in: vs[-1], 
                u_prev: us[-1],
                I_app: np.array(iapp).reshape([n, 1])}
        
        vo, uo, fired = sess.run([v_out, u_out, cond], feed_dict=feed)
        vs.append(vo)
        us.append(uo)
        
#plt.plot([np.asscalar(x) for x in vs])
allv = np.array(vs).reshape(2001, 1000)
spks = allv == 35
#plt.figure()
#plt.plot(wgs)

plt.figure()
inha, inhb = np.nonzero(spks[:,inh.reshape(1000)])
plt.plot(inha, inhb, 'r.')
exca, excb = np.nonzero(spks[:,exc.reshape(1000)])
plt.plot(exca, excb + inhb.max(), 'k.')
plt.axis([0, T, 0, n])
plt.title('Inhibitory and excitatory spikes')

plt.figure(figsize=(14,18))
for i, name in enumerate(var_names):

    if i < 8:
        plt.subplot(int("{}{}{}".format( int(np.ceil(len(var_names)/2)), 2, i+1)))
        plt.plot(stats[name])
        plt.title(name)

plt.show()
print("W:", W.mean(), "w_in:", w_in.mean())



