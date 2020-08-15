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
    y = dgp.random_walk(0, 1, 0, t)
    v = np.random.normal(0, 1, t)
    s = correlation_coefficient.corrcoef(y, v)
    plist.append(s)

plt.figure()
plt.hist(plist, bins=32, facecolor="w", edgecolor="black")
plt.title('Frequency distribution of the correlation coefficient between I(0) and I(1)')
plt.show()
