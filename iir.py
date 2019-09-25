import numpy as np
import matplotlib.pyplot as plt

N = 100
t = np.arange(0, N).T
t.shape = -1, 1
u = np.random.randn(100, 1)

alpha = 0.8
y = np.zeros((N, 1))
y[0] = u[0]

for k in range(1, 100):
    y[k] = alpha * y[k - 1] + (1 - alpha) * u[k - 1]

plt.plot(t, u, t, y)
plt.show()
