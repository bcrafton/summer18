import numpy as np

def action_to_str(action):
  return '{:+05.3f}'.format(action)

def to_str(actions):
  return [action_to_str(actions[0]), action_to_str(actions[2]) + " " + action_to_str(actions[3]), action_to_str(actions[1])]

def disp(x):
  for i in range(4):
    grid_str = ["", "", ""]
    for j in range(4):
      state = (4 - i - 1) * 4 + j
      state_str = to_str( x[state] )

      grid_str[0] += "         " + state_str[0]
      grid_str[1] += "   " + state_str[1]
      grid_str[2] += "         " + state_str[2]

    print (grid_str[0])
    print (grid_str[1])
    print (grid_str[2])
    print ("")
      
### SNN ###

snn = np.load('snn_weights.npy')
snn = snn.flatten()

avg_val = np.average( snn )
snn = snn / avg_val

max_val = np.max( np.absolute(snn) )
snn = snn / max_val

snn = np.round(snn, 3)

### Q ###

q = np.load('weights.npy')
# print("keras bias")
# print(q[1])
q = q[0]
q = q.flatten()

avg_val = np.average( q )
q = q / avg_val

max_val = np.max( np.absolute(q) )
q = q / max_val

q = np.round(q, 3)

### compare ###

print (snn)
print (q)
print (snn - q)

disp(snn.reshape(16,4))
# disp(q.reshape(16,4))
# disp(snn.reshape(16,4) - q.reshape(16,4))
