import numpy as np
import matplotlib.pyplot as plt

N = 64 # Signal size
k0 = 7 # Frequency
# x = np.exp(1j * 2 * np.pi * k0 / N * np.arrange(N))
x = np.cos(2 * np.pi * k0 / N * np.arange(N)) # real signal

X = np.array([])

for k in range(N):
    s = np.exp(1j * 2 * np.pi * k / N * np.arange(N))
    X = np.append(X, sum(x * np.conjugate(s)))

plt.plot(np.arange(N), abs(X))
plt.axis([0, N - 1, 0, N])

plt.show()