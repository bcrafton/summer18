
import numpy as np

x = [1,2,3,4]

x = [np.transpose(x)]
print (x)
print (np.repeat(x, 4, axis=0))

x = np.ones((1,16))
y = np.ones((16,4))
print (x.dot(y))

print (y * -1)
