import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math


n = 100
np.random.seed(572)
e = np.random.normal(0, 0.8, n)
u = np.random.normal(0, 0.1, n)
y = np.empty(n)
alpha = np.empty(n)
y[0] = e[0]
alpha[0] = u[0]
co = 0.1
for t in range(1, n):
    y[t] = co+alpha[t-1]+e[t]
    alpha[t] = alpha[t-1]+u[t]

time = np.linspace(0, n-1, n)
plt.figure(figsize=[10, 5])
plt.plot(time, y, time, alpha)
plt.xlabel('index')
plt.ylabel('y')
plt.show()

# standard Kalman filter approach
a, p, k, v = [np.ones(n) for _ in range(4)]
a[0], p[0], v[0] = y[0], 10000, 0


def fu(mypa):
    q, co = abs(mypa[0]), abs(mypa[1])
    z = w = 1
    likelihood = sigmae = 0
    for t in range(1, n):
        k[t] = (z*w*p[t-1])/(pow(z, 2)*p[t-1]+1)
        p[t] = pow(w, 2)*p[t-1]-w*z*k[t]*p[t-1]+q
        v[t] = y[t]-z*a[t-1]
        a[t] = co+w*a[t-1]+k[t]*v[t]
        sigmae += (pow(v[t], 2)/(pow(z, 2)*p[t-1]+1))
        likelihood += .5*math.log(2*math.pi)+.5+.5*math.log(pow(z, 2)*p[t-1]+1)
    return likelihood+.5*n*math.log(sigmae/n)


results = minimize(fu, [.6, .2])
z = w = 1
q, co = results.x[0], results.x[1]
sigmae = 0
for t in range(1, n):
    k[t] = (z*w*p[t-1])/(pow(z, 2)*p[t-1]+1)
    p[t] = pow(w, 2)*p[t-1]-w*z*k[t]*p[t-1]+q
    v[t] = y[t]-z*a[t-1]
    a[t] = co+w*a[t-1]+k[t]*v[t]
    sigmae = sigmae+pow(v[t], 2)/(pow(z, 2)*p[t-1]+1)
#This is the variance of e
print('sigmae = ', np.round(sigmae/(n-1), 4))
#This is the variance of u
print('sigmau = ', np.round(q*(sigmae/(n-1)), 4))


def generateTheta(n, sigmae, sigmau, co):
    e = np.random.normal(0, sigmae, n)
    u = np.random.normal(0, sigmau, n)
    y = np.empty(n)
    alpha = np.empty(n)
    y[0] = e[0]
    alpha[0] = u[0]
    for t in range(1, n):
        alpha[t] = co+alpha[t - 1] + u[t]
        y[t] = alpha[t-1]+e[t]
    return y


def EstimateTheta(y):
    n = len(y)
    a, p, k, v = [np.ones(n) for _ in range(4)]
    a[0], p[0] = 0, 0
    results = minimize(fu, [.5, .2])
    v[0] = 0
    z = w = 1
    q, co = results.x[0], results.x[1]
    sigmae = 0
    for t in range(1, n):
        k[t] = (z*w*p[t-1])/(pow(z, 2) * p[t-1]+1)
        p[t] = pow(w, 2)*p[t-1]-w*z*k[t]*p[t-1]+q
        v[t] = y[t]-z*a[t-1]
        a[t] = co+w*a[t-1]+k[t]*v[t]
        sigmae += pow(v[t], 2)/(pow(z, 2)*p[t-1]+1)
    sigmae = sigmae / len(y)
    sigmau = q * sigmae
    return [sigmae, sigmau, co]

np.random.seed(11)
print(EstimateTheta(generateTheta(100, .6, .2, 1)))
