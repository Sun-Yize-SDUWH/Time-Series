import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import arma


# read the data
data = pd.read_csv('../../data/nao.csv')
t = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d')
xt = np.array(data.iloc[:, 1])

step = 12
k = np.arange(1, step+1, 1)
r = np.zeros(step)
phi = np.zeros(step)
for i in range(1, step+1):
    r[i-1] = arma.autocorrelation_function(i, xt)
phi = np.diag(arma.partial_autocorrelation_function(r))

result = pd.DataFrame(np.array([k, r, phi]).transpose())
result.columns = ['k', 'r(k)', 'phi(kk)']
print(result)

ar_data = arma.autoregressive(phi[0:1], 0.994, xt[0:1], 0, len(xt))
ma_data = arma.autoregressive([0.175], 0.995, xt[0:1], 0, len(xt))

# plot
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t, xt)
plt.title('Observed')
plt.subplot(3, 1, 2)
plt.plot(t, ar_data)
plt.title('AR(1)')
plt.subplot(3, 1, 3)
plt.plot(t, ma_data)
plt.title('MA(1)')
plt.show()
