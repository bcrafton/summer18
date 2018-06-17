
import numpy as np

m1 = (1e5 - 5e4) / (0.9 - 0.0)
m2 = (1e6 - 1e5) / (1.0 - 0.9)
m3 = (8e5 - 1e6) / (0.1 - 1.0) 
m4 = (5e4 - 8e5) / (0.0 - 0.1) 

print (m1)
print (m1 * 0.1)
print (5e4)
print (m1 * 0.1 + 5e4)

def mott(r, v):
    if (v >= 0.0 and r >= 5e4 and r <= 1e5):
        r = m1 * v + 5e4
        print ("m1")
    elif (v > 0.9 and v <= 1.0 and r >= 1e5 and r <= 1e6):
        r = m2 * (v - 0.9) + 1e5
        print ("m2")
    elif (v > 0.1 and r > 8e5):
        r = 1e6 - m3 * v
        print ("m3")
    elif (v > 0.0 and r > 5e4):
        r = 8e5 - m4 * v
        print ("m4")
    else:
        print ("this should never happen")
        assert(False)
    return r, v
        
r = 5e4
v = 0.01
print (r, v)
for i in range(100):
    r, v = mott(r, v)
    v += 0.01
    v = np.clip(v, 0, 1.0)
    print (r, v)

for i in range(100):
    r, v = mott(r, v)
    v -= 0.01
    v = np.clip(v, 0, 1.0)
    print (r, v)
