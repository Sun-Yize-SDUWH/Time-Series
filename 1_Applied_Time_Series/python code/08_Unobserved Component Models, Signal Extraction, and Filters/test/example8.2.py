import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import hpfilter


data = pd.read_csv('../../data/global_temps.csv')
t = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d')
xt = data.iloc[:, 1]

filter1 = hpfilter.hpfilter(xt, 14400)
filter2 = hpfilter.hpfilter(xt, 129600)
filter3 = hpfilter.hpfilter(xt, 500000)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, xt)
plt.title('Observed')
plt.subplot(2, 1, 2)
plt.plot(t, filter1, '--b')
plt.plot(t, filter2, '--g')
plt.plot(t, filter3, 'r')
plt.title('hpfilter')
plt.legend(['lamb=14400', 'lamb=129600', 'lamb=500000'])
plt.show()
