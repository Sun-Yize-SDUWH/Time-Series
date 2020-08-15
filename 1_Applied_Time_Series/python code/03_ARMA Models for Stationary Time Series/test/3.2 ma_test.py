import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import arma


t = np.arange(100)

xt1 = arma.moving_average([0.8], 5, 0, 100)
xt2 = arma.moving_average([-0.8], 5, 0, 100)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, xt1)
plt.title("phi = 0.8")

plt.subplot(2, 1, 2)
plt.plot(t, xt2)
plt.title("phi = -0.8")
plt.show()
