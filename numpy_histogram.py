import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(1, 1000, (10000000, 10))
y = np.sum(x, axis=1)

plt.hist(y, 250)
plt.grid(True)
plt.show()
