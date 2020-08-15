import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import sys
sys.path.append("..")
import exp_smoothing


np.random.seed(2)
data = pd.read_csv('../../data/beer.csv')
t = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d')
y = data.iloc[:, 1]
xt = np.diff(data.iloc[:, 1])

h = 13
addho = exp_smoothing.add_holt(xt, 0.1, 0.79, 0.33, 4, h)
mulho = exp_smoothing.mul_holt(xt, 0.1, 0.81, 0, 4, h)

addho = exp_smoothing.returndiff(y[0], addho)
mulho = exp_smoothing.returndiff(y[0], mulho)

tt = pd.date_range('2017-10-01', periods=13, freq='3M')
plt.figure(figsize=[12, 15])
plt.subplot(3, 1, 1)
plt.plot(t, y)
plt.legend(['observed'])
plt.title('Obsrved')
plt.subplot(3, 1, 2)
plt.plot(t, addho[:-h], t, mulho[:-h])
plt.legend(['Holt add', 'Holt mul'])
plt.title('Compare')
plt.subplot(3, 1, 3)
plt.plot(tt, addho[-13:], tt, mulho[-13:])
plt.legend(['Holt add', 'Holt mul'])
plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y'))
plt.vlines(x='2017-12-31', ymin=5000, ymax=9200, colors='r', linestyles='--')
plt.vlines(x='2018-12-31', ymin=5000, ymax=9200, colors='r', linestyles='--')
plt.vlines(x='2019-12-31', ymin=5000, ymax=9200, colors='r', linestyles='--')
plt.title('Forecast')
plt.show()

