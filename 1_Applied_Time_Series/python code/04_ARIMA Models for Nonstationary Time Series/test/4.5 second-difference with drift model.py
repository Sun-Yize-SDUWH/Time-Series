import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import arima


xt = arima.autoregressive_integrated_moving_average(2, [1], [1], 3, [10, 10], 2, 100)
t = np.arange(100)

plt.figure()
plt.plot(t, xt)
plt.title('“second-difference with drift')
plt.show()
