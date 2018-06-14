
import numpy as np
import pylab as plt

m1 = (1e6 - 8e5) / (0.95 - 0.0)
m2 = (8e5 - 5e4) / (1.0 - 0.95)
m3 = (5e4 - 5e5) / (0.55 - 1.0)
m4 = (5e5 - 1e6) / (0.5 - 0.55)

def mott(v, state):

    if (state == 0 and v <= 1.0):
        r = 1e6 - m1 * v
        r = max(r, 8e5)
        r = min(r, 1e6)
        state = 0
        print ("m1")
        print (r)
    elif (state == 0 and v > 0.95):
        r = 8e5 - m2 * (v - 0.95);
        r = max(r, 5e4);
        r = min(r, 8e5);
        if (v > 1.0):
            state = 1;
        print ("m2")
    elif (state == 1 and v >= 0.55):
        r = 5e4 + m3 * (1.0 - v);
        r = max(r, 5e4);
        r = min(r, 5e5);
        state = 1;
        print ("m3")
    elif (state == 1 and v < 0.55):
        r = 5e5 + m4 * (0.55 - v);
        r = max(r, 5e5);
        r = min(r, 1e6);
        if (v < 0.5):
            state = 0;
        print ("m4")
        print (r)
    else:
        print ("this should never happen")
        
    return r, state
        
r = 1e6
state = 0
v = 0.25

rs = []
vs = []

for i in range(100):
    r, state = mott(v, state)
    
    vs.append(v)
    rs.append(r)
    
    v += 0.01
    v = np.clip(v, 0, 1.5)

for i in range(100):
    r, state = mott(v, state)
    
    vs.append(v)
    rs.append(r)
    
    v -= 0.01
    v = np.clip(v, 0, 1.5)


plt.plot(vs, rs)
plt.show()


