import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import hpfilter


data = pd.read_csv('../../data/gdp.csv')
t = np.arange(1270, 1914, 1)
y = data.iloc[:, 1]
xt = np.log(y)

lamb = 10000
filter1 = hpfilter.hpfilter(xt, lamb)
filter2 = hpfilter.hpfilter(y, lamb)
annual = pd.DataFrame(filter2).pct_change()

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, xt, t, filter1, '--')
plt.legend(['Observed', 'hpfilter'])
plt.title('hpfilter')
plt.subplot(2, 1, 2)
plt.plot(t[:], annual)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('yearly growth rate')
plt.show()
