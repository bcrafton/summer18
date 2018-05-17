
import numpy as np


x = np.load('snn_weights.npy')
x = x.reshape(16,4)
max_val = np.max( np.absolute(x) )
x = x / max_val
x = np.round(x, 3)

def action_to_str(action):
  return '{:+05.3f}'.format(action)

def to_str(actions):
  return [action_to_str(actions[0]), action_to_str(actions[2]) + " " + action_to_str(actions[3]), action_to_str(actions[1])]

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
    
# print '{:+05.3f}'.format(3.141592653589793)
# print ("%1.4f" % (1.058))
