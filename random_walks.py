import numpy as np
import matplotlib.pyplot as plt

x_0 = np.array([[0], [0]])
delta_x = np.random.normal(0, 1, (2,10))
X = np.concatenate((x_0, np.cumsum(delta_x, axis=1)), axis=1)
plt.plot(X[0], X[1], 'ro-')
plt.show()
