import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import smoothing

# read the data
data = pd.read_csv('../../data/beer.csv')
t = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d')
xt = np.array(data.iloc[:, 1])

# stationarity transformation
transform1 = smoothing.moving_average(xt, 5)

diff = xt[2:-2] - transform1
s1 = s2 = s3 = s4 = []
for i in range(len(diff)):
    if (i+2) % 4 == 0:
        s1 = np.append(s1, diff[i])
    elif (i + 2) % 4 == 1:
        s2 = np.append(s2, diff[i])
    elif (i + 2) % 4 == 2:
        s3 = np.append(s3, diff[i])
    else:
        s4 = np.append(s4, diff[i])
s1 = np.mean(s1)
s2 = np.mean(s2)
s3 = np.mean(s3)
s4 = np.mean(s4)
season = np.tile([s1, s2, s3, s4],  len(t)//4+1)
irregular = diff - season[2:len(diff)+2]


# plot
plt.figure(1)
plt.plot(t, xt, t[2:-2], transform1, '--')
plt.title('Observed and Trend')
plt.figure(2)
plt.plot(t, season[:len(t)])
plt.title('Seasonal')
plt.figure(3)
plt.plot(t[2:-2], irregular)
plt.title('Irregular')
plt.show()

