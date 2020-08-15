import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import stationarity


# read the data
data = pd.read_csv('../../data/wine_spirits.csv')
t = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d')
xt = data.iloc[:, 1]

# stationarity transformation
transform1 = stationarity.first_difference(xt)
transform2 = stationarity.second_differences(xt)

# plot
plt.figure(1)
plt.plot(t, xt)
plt.title('Observed')
plt.figure(2)
plt.plot(t[1:], transform1)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('first-difference')
plt.figure(3)
plt.plot(t[2:], transform2)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('second-differences')
plt.show()
