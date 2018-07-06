from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

x1 = np.genfromtxt('rst_vmem.csv',delimiter=',')
x2 = np.genfromtxt('rst_vo2.csv',delimiter=',')
x = np.transpose([x1, x2])
y = np.genfromtxt('rst_m12.csv',delimiter=',')

ax = plt.axes(projection='3d')
ax.plot3D(x1, x2, y)
plt.show()

'''
model = Sequential()
model.add(Dense(400, input_dim=2))
model.add(Dense(400))
model.add(Dense(1))
model.compile(loss='mean_absolute_percentage_error', optimizer=Adam(lr=0.0001))
model.fit(x, y, epochs=1000)

x = np.transpose([np.linspace(1, 1, 100), np.linspace(0, 1, 100)])
ret = model.predict(x)
plt.plot(np.linspace(0, 1, 100), ret)
plt.show()
'''
