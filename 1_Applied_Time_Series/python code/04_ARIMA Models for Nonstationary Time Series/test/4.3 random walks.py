import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import random_walk


n = 100
x0 = 10
sigma = 9
const = 2
t = np.arange(n)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, random_walk.random_walk(x0, sigma, 0, n))
plt.title('x(t)=x(t-1)+a(t), x0=10')
plt.subplot(2, 1, 2)
plt.plot(t, random_walk.random_walk(x0, sigma, const, n))
plt.title('x(t)=2+x(t-1)+a(t), x0=10')
plt.show()
