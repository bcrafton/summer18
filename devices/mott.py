
import numpy as np
import pylab as plt

m1 = (1e5 - 5e4) / (0.9 - 0.0)
m2 = (1e6 - 1e5) / (1.0 - 0.9)
m3 = (8e5 - 1e6) / (0.1 - 1.0) 
m4 = (5e4 - 8e5) / (0.0 - 0.1) 

def mott(r, v):
    global state
    if (v >= 0.0 and v <= 0.9 and state == 0):
        r = m1 * v + 5e4
    elif (v > 0.9 and v <= 1.0 and state == 0):
        r = m2 * (v - 0.9) + 1e5
        if (r >= 1e6):
            state = 1
        
    elif (v >= 0.1 and v <= 1.0 and state == 1):
        r = 1e6 - m3 * (0.9 - v)
    elif (v >= 0.0 and v < 0.1 and state == 1):
        r = 8e5 - m4 * (0.1 - v)
        if (r <= 5e4):
            state = 0
        
    else:
        print ("this should never happen")

    return r, v

vs = []
rs = []

state = 0
r = 5e4
v = 0.01

print (r, v)
vs.append(v)
rs.append(r)

for i in range(100):
    r, v = mott(r, v)
    
    vs.append(v)
    rs.append(r)
    print (r, v)
    
    v += 0.01
    v = np.clip(v, 0, 1.0)


for i in range(100):
    r, v = mott(r, v)
    
    vs.append(v)
    rs.append(r)
    print (r, v)
    
    v -= 0.01
    v = np.clip(v, 0, 1.0)
    
for i in range(100):
    r, v = mott(r, v)
    
    vs.append(v)
    rs.append(r)
    print (r, v)
    
    v += 0.01
    v = np.clip(v, 0, 1.0)
    
    
plt.plot(vs, rs)
plt.show()









