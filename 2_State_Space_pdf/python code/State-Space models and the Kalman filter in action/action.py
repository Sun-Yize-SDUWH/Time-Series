import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math


n = 100
np.random.seed(1256)
e = np.random.normal(0, 0.1, n)
u = np.random.normal(0, 0.05, n)
const = 0.2
y = np.empty(n)
alpha = np.empty(n)
y[0] = e[0]
alpha[0] = u[0]

for t in range(1, n):
    y[t] = alpha[t-1]+e[t]
    alpha[t] = const + 0.85*alpha[t-1]+u[t]

time = np.linspace(0, n-1, n)
plt.figure(figsize=[10, 5])
plt.plot(time, alpha)
plt.scatter(time, y, facecolors='none', edgecolors='b')
plt.xlabel('index')
plt.ylabel('y')
plt.show()


# standard Kalman filter approach
a, p, k, v = [np.ones(n) for _ in range(4)]
z, a[0], p[0] = 1, 0, 1


def fu(mypa):
    w, q, co = abs(mypa[0]), abs(mypa[1]), abs(mypa[2])
    likelihood, sigmae = 0, 0
    for t in range(1, n):
        k[t] = (z*w*p[t-1])/(pow(z, 2)*p[t-1]+1)
        p[t] = pow(w, 2)*p[t-1]-w*z*k[t]*p[t-1] + q
        v[t] = y[t]-z*a[t-1]
        a[t] = const+w*a[t-1]+k[t]*v[t]
        sigmae += (pow(v[t], 2)/(pow(z, 2)*p[t-1]+1))
        likelihood += .5*math.log(2*math.pi)+.5+.5*math.log(pow(z, 2)*p[t-1]+1)
    return likelihood+.5*n*math.log(sigmae/n)


results = minimize(fu, [.9, 1, .1])
v[0] = 0
w, q, co = abs(results.x[0]), abs(results.x[1]), abs(results.x[2])
likelihood, sigmae = 0, 0

for t in range(1, len(y)):
    k[t] = (z*w*p[t-1])/(pow(z, 2)*p[t-1]+1)
    p[t] = pow(w, 2)*p[t-1]-w*z*k[t]*p[t-1]+q
    v[t] = y[t]-z*a[t-1]
    a[t] = co+w*a[t-1]+k[t]*v[t]
    sigmae = sigmae+pow(v[t], 2)/(pow(z, 2)*p[t-1]+1)
    likelihood = likelihood+.5*math.log(2*math.pi)+.5+.5*math.log(pow(z, 2)*p[t-1]+1)
likelihood<-likelihood+.5*n*math.log(sigmae/n)
sigmae = sigmae/len(y)
sigmau = q*sigmae

print('co,w,z,sigmae,sigmau', '\n', co, w, z, sigmae, sigmau)
