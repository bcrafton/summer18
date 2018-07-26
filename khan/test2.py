
import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(52e-3, 65e-3, 1000)
y1 = 17e-3 - x1

x2 = np.linspace(-65e-3, -52e-3, 1000)
y2 = -100e-3 - x2

x1 = np.linspace(40e-3, 60e-3, 1000)
y1 = 15e-3 - x1

x2 = np.linspace(-60e-3, -40e-3, 1000)
y2 = -85e-3 - x2

print np.max(y1), np.max(y2)
print np.min(y1), np.min(y2)
print y1 == y2
print np.array_equal(y1, y2)

plt.plot(x1, y1, x2, y2)
plt.show()
