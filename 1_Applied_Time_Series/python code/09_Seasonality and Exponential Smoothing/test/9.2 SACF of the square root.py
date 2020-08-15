import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import Seasonality


data = pd.read_csv('../../data/rainfall.csv')
t = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d')
xt = data.iloc[:, 1]

xt = np.sqrt(xt)
r = np.ones(48)
for k in range(1, 49):
    r[k-1] = Seasonality.autocorrelation_function(k, xt)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, xt)
plt.subplot(2, 1, 2)
plt.grid(True)
plt.stem(r, use_line_collection=True)
plt.show()
