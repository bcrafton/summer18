
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
    def __init__(self, size):
        self.size = size
        self.w =  np.random.uniform(0.0, 1.0, size=(self.size)) * 2 * 0.12 - 0.12
        self.y = 0.0
        self.dy = 0.0

        self.forward_step = 0
        self.backward_step = 0

    def forward(self, x):
        assert(self.backward_step == 0 or self.backward_step == self.size)
        self.backward_step = 0
    
        self.y += self.w[self.forward_step] * x
        self.forward_step += 1
        
        if self.forward_step == self.size:
            self.y = relu(self.y)
            self.dy = drelu(self.y)
            
        return self.y
    
    def backward(self, x, e):
        assert(self.forward_step == 0 or self.forward_step == self.size)
        self.forward_step = 0
    
        dw = x * e * self.dy
        self.w[self.backward_step] += dw
        
        self.backward_step += 1
        
                
