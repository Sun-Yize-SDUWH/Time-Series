import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import Seasonality


data = pd.read_csv('../../data/beer.csv')
t = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d')
xt = data.iloc[:, 1]

xt = Seasonality.first_difference(xt)
diff = np.ones(len(xt)-4)
for i in range(4, len(xt)):
    diff[i-4] = xt[i] - xt[i-4]

r = np.ones(24)
for k in range(1, 25):
    r[k-1] = Seasonality.autocorrelation_function(k, diff)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t[1:], xt)
plt.subplot(2, 1, 2)
plt.grid(True)
plt.stem(r, use_line_collection=True)
plt.show()
