import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def first_difference(xt):
    y = np.array([])
    for i in range(len(xt)-1):
        y = np.append(y, xt[i+1] - xt[i])
    return y


def autocorrelation_function(k, xt):
    mean = np.mean(xt)
    var = np.var(xt)
    temp = 0
    for i in range(k, len(xt)):
        temp += (xt[i] - mean) * (xt[i - k] - mean)
    r = temp / (len(xt) * var)
    return r


def partial_autocorrelation_function(acf):
    k = len(acf)
    phi = np.empty([k, k])
    for i in range(k):
        if i == 0:
            phi[0][0] = acf[0]
        elif i == 1:
            phi[1][1] = (acf[1]-np.power(acf[0], 2))/(1-np.power(acf[0], 2))
        else:
            temp1 = temp2 = 0
            for j in range(i):
                if i-1 != j:
                    phi[i-1][j] = phi_calculate(phi, [i-1, j])
                temp1 += phi[i-1][j]*acf[i-j]
                temp2 += phi[i-1][j]*acf[j]
            phi[i][i] = (acf[i] - temp1) / (1 - temp2)
    return phi


def phi_calculate(phi, s):
    result = phi[s[0]-1, s[1]] - phi[s[0], s[0]]*phi[s[0]-1, s[0]-s[1]-1]
    return result


# read the data
data = pd.read_csv('../../data/interest_rates.csv')
t = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d')[1:]
xt = first_difference(data.iloc[:, 1])

step = 12
k = np.arange(1, step+1, 1)
r = np.zeros(step)
phi = np.zeros(step)
for i in range(1, step+1):
    r[i-1] = autocorrelation_function(i, xt)
phi = np.diag(partial_autocorrelation_function(r))

result = pd.DataFrame(np.array([k, r, phi]).transpose())
result.columns = ['k', 'r(k)', 'phi(kk)']
print(result)

plt.figure()
plt.plot(t, xt)
plt.title('first-difference')
plt.show()
