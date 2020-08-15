import numpy as np
import matplotlib.pyplot as plt


np.random.seed(7)
n = 103
e = np.random.normal(0, 0.5, n)
u = np.random.normal(0, 0.4, n)
y = np.empty(n)
alpha = np.empty(n)
factor = [1.7, .3, 1.9, .1]
seasonal = np.tile(factor, n // 4+1)[:n]
y[0] = e[0]
alpha[0] = 5 + u[0]
for t in range(1, n):
    y[t] = seasonal[t]*(alpha[t-1]+e[t])
    alpha[t] = alpha[t-1]+u[t]

time = np.linspace(0, n-1, n)
plt.figure(figsize=[10, 5])
plt.plot(time, y, time, alpha)
plt.xlabel('index')
plt.ylabel('y')
plt.title('y and alpha')
plt.show()


s = 4
n = len(y)
w = np.tile(1/(2*s), s+1)
w[1:s] = 1/s
cma = np.full(len(y), np.nan)
for g in range(len(y)-s):
    cma[int(g+s/2)] = np.sum(w*y[g:(g+s+1)])
residuals = y/cma
sfactors = np.empty(s)
for seas in range(s):
    temp = np.array([])
    sfactors[seas] = np.nanmean(residuals[slice(seas, len(y)-s+seas+1, s)])
sfactors = sfactors*s/np.sum(sfactors)
newseries = y/np.tile(sfactors, n//s+1)[:n]


np.random.seed(7)
n = 103
e = np.random.normal(0, 0.5, n)
u = np.random.normal(0, 0.4, n)
y = np.empty(n)
alpha = np.empty(n)
seasfactor = [1.7, .3, 1.9, .1]
seasonal = np.tile(seasfactor, n // 4+1)[:n]
y[0] = e[0]
alpha[0] = 5 + u[0]
for t in range(1, n):
    y[t] = seasonal[t]*(alpha[t-1]+e[t])
    alpha[t] = alpha[t-1]+u[t]

time = np.linspace(0, n-1, n)
plt.figure(figsize=[10, 5])
plt.plot(time, y, time, alpha)
plt.xlabel('index')
plt.ylabel('y')
plt.title('y and alpha')
plt.show()


s = 4
n = len(y)
w = np.tile(1/(2*s), s+1)
w[1:s] = 1/s
cma = np.full(len(y), np.nan)
for g in range(len(y)-s):
    cma[int(g+s/2)] = np.sum(w*y[g:(g+s+1)])
residuals = y/cma
sfactors = np.empty(s)
for seas in range(s):
    temp = np.array([])
    sfactors[seas] = np.nanmean(residuals[slice(seas, len(y)-s+seas+1, s)])
sfactors = sfactors*s/np.sum(sfactors)
newseries = y/np.tile(sfactors, n//s+1)[:n]


time = np.linspace(0, n-1, n)
plt.figure(figsize=[10, 5])
plt.plot()
plt.plot(time, newseries, time, alpha+e, '--')
plt.ylabel('combine(newseries, alpha+e)')
plt.title('newseries and alpha+e')
plt.show()

print('factor = ', '\n', factor)
print('sfactors = ', '\n', np.round(sfactors, 2))
