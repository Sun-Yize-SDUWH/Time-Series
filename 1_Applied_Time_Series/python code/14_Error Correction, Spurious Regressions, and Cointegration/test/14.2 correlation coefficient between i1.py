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
    x = dgp.random_walk(0, 1, 0, t)
    y = dgp.random_walk(0, 1, 0, t)
    s = correlation_coefficient.corrcoef(x, y)
    plist.append(s)

plt.figure()
plt.hist(plist, bins=40, facecolor="w", edgecolor="black")
plt.title('Frequency distribution of the correlation coefficient between I(1) series.')
plt.show()
