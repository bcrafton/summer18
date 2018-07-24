from brian2 import *
import matplotlib.cm as cmap

start_scope()

eqs = '''
dv/dt = (I-v)/tau : 1
I : 1
tau : second
'''
A = NeuronGroup(1, eqs, threshold='v>1', reset='v = 0', method='exact')
B = NeuronGroup(2, eqs, threshold='v>1', reset='v = 0', method='exact')

A.tau = [10]*ms
B.tau = [100, 100]*ms
A.I = [2]
B.I = [0, 0]

S = Synapses(A, B, 'w : 1', on_pre='v_post += w')
S.connect()
S.delay = '0.1*ms + rand() * 1.9*ms'
S.w = '0.2'

print S.delay

M = StateMonitor(B, 'v', record=True)

run(50*ms)

print S.delay

plt.plot(M.t/ms, M.v[0], label='Neuron 0')
plt.plot(M.t/ms, M.v[1], label='Neuron 1')
plt.show()
