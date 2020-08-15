import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math


np.random.seed(4241)
n, h, co = 1293, 6, 0.03
e = np.random.normal(0, 0.4, n)
u = np.random.normal(0, 0.1, n)
my = np.empty(n)
alpha = np.empty(n)
factor = [.3, .9, 1.3, 1.5]
seasonal = np.tile(factor, n // 4+1)[:n]
my[0] = e[0]
alpha[0] = u[0]
for t in range(1, n):
    my[t] = seasonal[t]*(alpha[t-1]+e[t])
    alpha[t] = co+alpha[t-1]+u[t]
yy = my[300:398]

time = np.linspace(0, len(yy)-1, len(yy))
plt.figure(figsize=[10, 5])
plt.plot(time, yy)
plt.xlabel('index')
plt.ylabel('yy')
plt.title('yy')
plt.show()

y = yy[:len(yy)-h]
s, n = 4, len(y)
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


# standard Kalman filter approach
a, p, k, v = [np.ones(n) for _ in range(4)]
a[0], p[0], v[0] = newseries[0], 10000, 0

def funcTheta(parameters):
    q, co = abs(parameters[0]), abs(parameters[1])
    z = w = 1
    likelihood = sigmae = 0
    for t in range(1, len(newseries)):
        k[t] = (z*w*p[t-1])/(pow(z, 2)*p[t-1]+1)
        p[t] = pow(w, 2)*p[t-1]-w*z*k[t]*p[t-1]+q
        v[t] = newseries[t]-z*a[t-1]
        a[t] = co+w*a[t-1]+k[t]*v[t]
        sigmae += (pow(v[t], 2)/(pow(z, 2)*p[t-1]+1))
        likelihood += .5*math.log(2*math.pi)+.5+.5*math.log(pow(z, 2)*p[t-1]+1)
    return likelihood+.5*n*math.log(sigmae/n)


results = minimize(funcTheta, [.6, .2])
z = w = 1
q, co, sigmae = results.x[0], results.x[1], 0

for t in range(1, len(newseries)):
    k[t] = (z*w*p[t-1])/(pow(z, 2)*p[t-1]+1)
    p[t] = pow(w, 2)*p[t-1]-w*z*k[t]*p[t-1]+q
    v[t] = newseries[t]-z*a[t-1]
    a[t] = co+w*a[t-1]+k[t]*v[t]
    sigmae += pow(v[t], 2)/(pow(z, 2)*p[t-1]+1)

#This is the drift parameter
print('co = ', np.round(co, 4))
#This is the variance of e
print('sigmae = ', np.round(sigmae/(n-1), 4))
#This is the variance of u
print('sigmau = ', np.round(q*(sigmae/(n-1)), 4))

sfactnh = np.tile(sfactors, int((n+h)/s+1))[0:n+h]
sfactout = sfactnh[len(sfactnh)-h:len(sfactnh)]
w = z = 1
MyForecasts = np.array([])
MyForecasts = np.append(MyForecasts, a[len(newseries)-1])
for o in range(1, h):
    MyForecasts = np.append(MyForecasts, co+MyForecasts[o-1])
SeasonalForecast = MyForecasts*sfactout

time = np.linspace(0, h-1, h)
plt.figure()
plt.plot(time, yy[len(yy)-h:len(yy)], time, SeasonalForecast, '--r')
plt.xlabel('index')
plt.ylabel('yy and forecast')
plt.title('black is y_t, red is the forecasts')
plt.show()
