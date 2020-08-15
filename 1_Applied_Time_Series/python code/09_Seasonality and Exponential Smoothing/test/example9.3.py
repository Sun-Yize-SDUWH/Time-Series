import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import exp_smoothing


data = pd.read_csv('../../data/global_temps.csv')
t = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d')[1:]
xt = np.diff(data.iloc[:, 1])

ssmo = exp_smoothing.single_exponential_smoothing(xt, 0.45)
dsmo = exp_smoothing.double_exponential_smoothing(xt, 0.196)

RMSE1 = np.sqrt(np.sum(np.power(xt[:]-ssmo[:], 2))/len(xt))
RMSE2 = np.sqrt(np.sum(np.power(xt[:]-dsmo[:], 2))/len(xt))

print('single exp RMSE = ', RMSE1, '\n', 'double exp RMSE = ', RMSE2)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, xt, t, ssmo)
plt.legend(['observed', 'ssmo'])
plt.title('single exp smoothing')
plt.subplot(2, 1, 2)
plt.plot(t, xt, t, dsmo)
plt.legend(['observed', 'dsmo'])
plt.title('double exp smoothing')
plt.show()
