import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math


n = 100
np.random.seed(153)
e = np.random.normal(0, 0.5, n)
u = np.random.normal(0, 0.2, n)
y = np.empty(n)
alpha = np.empty(n)
y[0] = e[0]
alpha[0] = u[0]

for t in range(1, n):
    y[t] = alpha[t-1]+e[t]
    alpha[t] = alpha[t-1]+u[t]

time = np.linspace(0, n-1, n)
plt.figure(figsize=[10, 5])
plt.plot(time, alpha)
plt.scatter(time, y, facecolors='none', edgecolors='b')
plt.xlabel('index')
plt.ylabel('y')
plt.title('y and alpha')
plt.show()

# standard Kalman filter approach
a, p, k, v = [np.ones(n) for _ in range(4)]
a[0], p[0] = y[0], 10000

def fu(mypa):
    q = abs(mypa)
    z = w = 1
    likelihood = sigmae = 0
    for t in range(1, n):
        k[t] = (z*w*p[t-1])/(pow(z, 2)*p[t-1]+1)
        p[t] = pow(w, 2)*p[t-1]-w*z*k[t]*p[t-1]+q
        v[t] = y[t]-z*a[t-1]
        a[t] = w*a[t-1]+k[t]*v[t]
        sigmae += (pow(v[t], 2)/(pow(z, 2)*p[t-1]+1))
        likelihood += .5*math.log(2*math.pi)+.5+.5*math.log(pow(z, 2)*p[t-1]+1)
    return likelihood+.5*n*math.log(sigmae/n)


bnds = [[0, 1]]
results = minimize(fu, [0.2], bounds=bnds)
print('q = ', results.x[0])

z = w = 1
q = results.x[0]
sigmae = 0
for t in range(1, n):
    k[t] = (z*w*p[t-1])/(pow(z, 2)*p[t-1]+1)
    p[t] = pow(w, 2)*p[t-1]-w*z*k[t]*p[t-1]+q
    v[t] = y[t]-z*a[t-1]
    a[t] = w*a[t-1]+k[t]*v[t]
    sigmae = sigmae+pow(v[t], 2)/(pow(z, 2)*p[t-1]+1)
#This is the variance of e
print('sigmae = ', sigmae/(n-1))
#This is the variance of u
print('sigmau = ', q*(sigmae/(n-1)))
