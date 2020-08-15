import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1213)
n = 102
e = np.random.normal(0, 0.5, n)
u = np.random.normal(0, 0.1, n)
y = np.empty(n)
alpha = np.empty(n)
seasfactor = [5, -4, 2, -3]
s = 4
seasonal = np.tile(seasfactor, n // 4+1)[:n]
y[0] = e[0]+seasonal[1]
alpha[0] = u[0]
for t in range(1, n):
    y[t] = seasonal[t]+alpha[t-1]+e[t]
    alpha[t] = alpha[t-1]+u[t]

time = np.linspace(0, n-1, n)
plt.figure(figsize=[10, 5])
plt.plot(time, y, time, alpha)
plt.xlabel('index')
plt.ylabel('y')
plt.title('y and alpha')
plt.show()


import pandas as pd


y = np.array([6, 2, 1, 3, 7, 3, 2, 4])
cma = np.full(len(y), np.nan)
cma[2] = (.5*y[0]+y[1]+y[2]+y[3]+.5*y[4])/4
cma[3] = (.5*y[1]+y[2]+y[3]+y[4]+.5*y[5])/4
cma[4] = (.5*y[2]+y[3]+y[4]+y[5]+.5*y[6])/4
cma[5] = (.5*y[3]+y[4]+y[5]+y[6]+.5*y[7])/4
residuals = y-cma
result = pd.DataFrame(np.array([y, cma, residuals]).transpose())
result.columns = ['y', 'cma', 'residuals']
print(result)

factors = [np.nanmean([residuals[0], residuals[4]]), np.nanmean([residuals[1], residuals[5]]),
           np.nanmean([residuals[2], residuals[6]]), np.nanmean([residuals[3], residuals[7]])]
newseries = y - np.tile(factors, 2)

time = np.linspace(0, 7, 8)
plt.figure()
plt.plot()
plt.plot(time, y, time, newseries)
plt.ylabel('combine(y, newseries)')
plt.show()


# s is my frequency (for example: quarterly=4;monthly=12;weekly=52)
s = 4
n = len(y)
# This create the weights to be used in the moving average
w = np.tile(1/(2*s), s+1)
w[1:s] = 1/s
# This create the centered moving average vector
cma = np.full(len(y), np.nan)
# This calculate the centered moving averag
for g in range(len(y)-s):
    cma[int(g+s/2)] = np.sum(w*y[g:(g+s+1)])
# This is the residuals
residuals = y - cma
# this creates the s factors as we want
factors = np.empty(s)
for seas in range(s):
    temp = np.array([])
    factors[seas] = np.nanmean(residuals[slice(seas, len(y)-s+seas+1, s)])
# This allows to demean the factors variable
factors = factors - np.tile(np.mean(factors), s)
# this is the last step: we take out the seasonal component
newseries = y - np.tile(factors, n//s+1)[:n]

time = np.linspace(0, n-1, n)
plt.figure()
plt.plot()
plt.plot(time, y, time, newseries)
plt.ylabel('combine(y, newseries)')
plt.show()


np.random.seed(243)
n = 87
e = np.random.normal(0, 0.3, n)
u = np.random.normal(0, 0.1, n)
y = np.empty(n)
alpha = np.empty(n)
seasfactor = [5, -4, 2, -3]
s = 4
seasonal = np.tile(seasfactor, n//s+1)[:n]
y[0] = e[0]+seasonal[0]
alpha[0] = u[0]
for t in range(1,n):
    y[t] = seasonal[t]+alpha[t-1]+e[t]
    alpha[t] = alpha[t-1]+u[t]
w = np.tile(1/(2*s), s+1)
w[1:s] = 1/s
cma = np.full(len(y), np.nan)
for g in range(len(y)-s):
    cma[int(g+s/2)] = np.sum(w*y[g:(g+s+1)])
residuals = y - cma
factors = np.empty(s)
for seas in range(s):
    temp = np.array([])
    factors[seas] = np.nanmean(residuals[slice(seas, len(y)-s+seas+1, s)])
factors = factors - np.tile(np.mean(factors), s)
newseries = y - np.tile(factors, n//s+1)[:n]

time = np.linspace(0, n-1, n)
plt.figure(figsize=[10, 5])
plt.plot()
plt.plot(time, newseries, time, alpha+e, '--')
plt.ylabel('combine(newseries, alpha+e)')
plt.title('newseries and alpha+e')
plt.show()

print('factor = ', '\n', np.round(factors, 2))
print('seasfactor = ', '\n', seasfactor)
