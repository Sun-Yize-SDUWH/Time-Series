import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import arma


t = np.arange(100)

xt1 = arma.autoregressive([0.5], 5, [0], 0, 100)
xt2 = arma.autoregressive([-0.5], 5, [0], 0, 100)

plt.figure()
plt.subplot(2, 2, 1)
plt.grid(True)
plt.stem(np.power(0.5, np.arange(12)), use_line_collection=True)
plt.title("phi = 0.5")
plt.subplot(2, 2, 2)
plt.plot(t, xt1)
plt.title("phi = 0.5, x0 = 0")
plt.subplot(2, 2, 3)
plt.grid(True)
plt.stem(np.power(-0.5, np.arange(12)), use_line_collection=True)
plt.title("phi = -0.5")
plt.subplot(2, 2, 4)
plt.plot(t, xt2)
plt.title("phi = -0.5, x0 = 0")
plt.show()
