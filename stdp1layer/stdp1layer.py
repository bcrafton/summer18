from brian2 import *
import numpy as np
import cPickle, gzip, os, sys, signal

def load_training_set():
  global training_set, training_labels
  f = gzip.open('mnist.pkl.gz', 'rb')
  train, valid, test = cPickle.load(f)

  [training_set, training_labels] = train
  [validation_set, validation_labels] = valid
  [testing_set, testing_labels] = test

  for i in range( len(training_set) ):
    training_set[i] = training_set[i].reshape(28*28)
    training_set[i] = training_set[i] / np.max(training_set[i])
    training_set[i] = training_set[i]

  f.close()

def spike_plot(spike_mon):
  spike_plot_x = []
  spike_plot_y = []
  trains = spike_mon.spike_trains()
  for i in trains:
    spike_plot_x.extend(trains[i])

    sz = len(trains[i])
    vals = [i] * sz
    spike_plot_y.extend( vals )
  
  return spike_plot_x, spike_plot_y

taum = 10*ms
taupre = 20*ms
taupost = 20*ms

Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
taue = 5*ms

F = 15*Hz
gmax = .01

dApre = .01
dApost = -dApre * (taupre / taupost) * 1.05
dApost *= gmax
dApre *= gmax

eqs_neurons = '''
dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
dge/dt = -ge / taue : 1
'''

input = PoissonGroup(28*28, rates=F)
spike_mon0 = SpikeMonitor(input)
neurons = NeuronGroup(12*12, eqs_neurons, threshold='v>vt', reset='v = vr', method='exact')

###########################

S0 = Synapses(input, neurons,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''
                    ge += w
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)''',
             on_post='''
                    Apost += dApost
                    w = clip(w + Apre, 0, gmax)''',
             )
S0.connect()
S0.w = 'rand() * gmax'

# pretty sure [0,1] is the range of the weight to record.
# mon = StateMonitor(S0, 'w', record=[0, 1])
# NOOO [0,1] was actually record index 0 and 1...
# so we were just looking at indexes 0 and 1.

mon0 = StateMonitor(S0, 'w', record=range(12*12))
spike_mon1 = SpikeMonitor(neurons)

###########################

load_training_set()
for i in range(1000):
  input.rates = training_set[i] * F
  run(0.25 * second, report='text')

###########################

spike_plot_x0, spike_plot_y0 = spike_plot(spike_mon0)
spike_plot_x1, spike_plot_y1 = spike_plot(spike_mon1)

subplot(511)
plot(spike_plot_x0, spike_plot_y0, '.k')

subplot(512)
plot(mon0.t/second, mon0.w.T/gmax)
xlabel('Time (s)')
ylabel('Weight / gmax')

subplot(513)
plot(spike_plot_x1, spike_plot_y1, '.k')

tight_layout()
show()




