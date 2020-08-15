import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")


data = pd.read_csv('../../data/ftse.csv')
t = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d')
xt = data.iloc[:, 1]

plt.figure()
plt.plot(t, xt, t, arima)
plt.title('Observed')
plt.show()
