import numpy as np
import matplotlib.pyplot as plt


n = 100
t = np.arange(n)
a = np.random.normal(0, 9, n)
xt1 = 10 + 2 * t + a
xt2 = 10 + 5 * t - 0.03 * np.power(t, 2) + a

plt.figure()
plt.plot(t, xt1, t, xt2, '--')
plt.title('Simulated linear and quadratic trends')
plt.show()
