import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math


np.random.seed(134)
n = 105
e = np.random.normal(0, 0.5, n)
u = np.random.normal(0, 0.1, n)
y = np.empty(n)
alpha = np.empty(n)
co, y[0], alpha[0] = 0.06, e[0], u[0]

for t in range(1, n):
    alpha[t] = co+alpha[t - 1] + u[t]
    y[t] = alpha[t-1]+e[t]

time = np.linspace(0, n-1, n)
plt.figure(figsize=[10, 5])
plt.plot(time, alpha)
plt.xlabel('index')
plt.ylabel('y')
plt.show()

# I define x the variable with 100 observations.
n = 100
x = y[:]
a, p, k, v = [np.ones(n) for _ in range(4)]
a[0], p[0], v[0] = x[1], 10000, 0


def funcTheta(parameters):
    q, co = abs(parameters[0]), abs(parameters[1])
    z = w = 1
    likelihood = sigmae = 0
    for t in range(1, n):
        k[t] = (z*w*p[t-1])/(pow(z, 2)*p[t-1]+1)
        p[t] = pow(w, 2)*p[t-1]-w*z*k[t]*p[t-1]+q
        v[t] = x[t]-z*a[t-1]
        a[t] = co+w*a[t-1]+k[t]*v[t]
        sigmae += (pow(v[t], 2)/(pow(z, 2)*p[t-1]+1))
        likelihood += .5*math.log(2*math.pi)+.5+.5*math.log(pow(z, 2)*p[t-1]+1)
    return likelihood+.5*n*math.log(sigmae/n)


results = minimize(funcTheta, [.6, .2])
z = w = 1
q, co, sigmae = results.x[0], results.x[1], 0

for t in range(1, n):
    k[t] = (z*w*p[t-1])/(pow(z, 2)*p[t-1]+1)
    p[t] = pow(w, 2)*p[t-1]-w*z*k[t]*p[t-1]+q
    v[t] = x[t]-z*a[t-1]
    a[t] = co+w*a[t-1]+k[t]*v[t]
    sigmae += pow(v[t], 2)/(pow(z, 2)*p[t-1]+1)

#This is the drift parameter
print('co = ', np.round(co, 4))
#This is the variance of e
print('sigmae = ', np.round(sigmae/(n-1), 4))
#This is the variance of u
print('sigmau = ', np.round(q*(sigmae/(n-1)), 4))

t = 5
MyForecasts = np.ones(t)
# This is my one-step ahead for x:
MyForecasts[0] = a[n-1]
MyForecasts[1] = co+MyForecasts[0]
MyForecasts[2] = co+MyForecasts[1]
MyForecasts[3] = co+MyForecasts[2]
MyForecasts[4] = co+MyForecasts[3]

time = np.linspace(0, t-1, t)
plt.figure()
plt.plot(time, y[99:104], time, MyForecasts)
plt.xlabel('index')
plt.ylabel('y')
plt.show()
