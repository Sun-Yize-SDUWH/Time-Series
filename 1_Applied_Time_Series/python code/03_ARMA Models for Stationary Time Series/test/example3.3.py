import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import arma

# read the data
data = pd.read_csv('../../data/sunspots.csv')
t = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d')
xt = data.iloc[:, 1]

step = 9
k = np.arange(1, step+1, 1)
r = np.zeros(step)
phi = np.zeros(step)
for i in range(1, step+1):
    r[i-1] = arma.autocorrelation_function(i, xt)
phi = np.diag(arma.partial_autocorrelation_function(r))

result = pd.DataFrame(np.array([k, r, phi]).transpose())
result.columns = ['k', 'r(k)', 'phi(kk)']
print(result)

# plot
plt.figure()
plt.plot(t, xt)
plt.title('Observed')
plt.show()
