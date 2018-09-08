
import numpy as np

def relu(x):
    return (x > 0.0) * x
    
def drelu(x):
    return (x > 0.0)

'''
so there are 2 ways you can do this and one way is clearly better
1. all w's get a different x, sum(wx) = y, same e same y
2. all w's get the same x, we have to do a sum outside the core from all cores. all e's have to be sent to the cores as well 
'''

class NeuroCore:
    def __init__(self, size, last):
        self.size = size
        self.last = last
        self.w =  np.random.uniform(0.0, 1.0, size=(self.size)) * 2 * 0.12 - 0.12
        self.y = 0.0
        self.dy = 0.0

    def forward(self, x):
        self.y = np.sum(self.w * x)
        self.y = relu(self.y)
        
        if self.last:
            self.dy = 1.0
        else:
            self.dy = drelu(self.y)
            
        return self.y
    
    def backward(self, x, e):
        dw = x * e * self.dy
        self.w -= dw        
