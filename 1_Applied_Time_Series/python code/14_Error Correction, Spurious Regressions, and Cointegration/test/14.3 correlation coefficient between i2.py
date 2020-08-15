import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import correlation_coefficient
import dgp


np.random.seed(2)
n = 1000
t = 100
plist = []
for i in range(n):
    z = dgp.autoregressive([2, -1], 1, [0, 0], 0, t)
    w = dgp.autoregressive([2, -1], 1, [0, 0], 0, t)
    s = correlation_coefficient.corrcoef(z, w)
    plist.append(s)

plt.figure()
plt.hist(plist, bins=40, facecolor="w", edgecolor="black")
plt.title('Frequency distribution of the correlation coefficient between I(2) series.')
plt.show()
