import numpy as np

snn = np.load('snn_weights.npy')
snn = snn.flatten()
max_val = np.max( np.absolute(snn) )
snn = snn / max_val
snn = np.round(snn, 3)

q = np.load('weights.npy')
q = q[0]
q = q.flatten()
max_val = np.max( np.absolute(q) )
q = q / max_val
q = np.round(q, 3)

print snn
print q
print snn - q
