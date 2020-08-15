import numpy as np
from scipy.optimize import minimize
import math


n = 100
np.random.seed(61)
su, se = 0.1, 0.4
qreal = su/se
e = np.random.normal(0, se, n)
u = np.random.normal(0, su, n)
z, wreal = 1, 0.97
y = np.ones(n)
alpha = np.ones(n)
y[0], alpha[0] = e[0], u[0]
for t in range(1, n):
    y[t] = z*alpha[t-1]+e[t]
    alpha[t] = wreal*alpha[t-1]+u[t]


# standard Kalman filter approach
a, p, k, v = [np.ones(n) for _ in range(4)]
a[0], p[0] = 0, 10


def fu(mypa):
    w, q = abs(mypa[0]), abs(mypa[1])
    z, likelihood, sigmae = 1, 0, 0
    for t in range(1, n):
        k[t] = (z*w*p[t-1])/(pow(z, 2)*p[t-1]+1)
        p[t] = pow(w, 2)*p[t-1]-w*z*k[t]*p[t-1]+q
        v[t] = y[t]-z*a[t-1]
        a[t] = w*a[t-1]+k[t]*v[t]
        sigmae += (pow(v[t], 2)/(pow(z, 2)*p[t-1]+1))
        likelihood += .5*math.log(2*math.pi)+.5+.5*math.log(pow(z, 2)*p[t-1]+1)
    return likelihood+.5*n*math.log(sigmae/n)


results = minimize(fu, [.85, .5])
print("The results of the standard KF approach", '\n', results.x)
print("The true parameters", '\n', [wreal, qreal])
