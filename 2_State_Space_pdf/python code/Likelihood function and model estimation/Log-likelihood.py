import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math


n = 100
np.random.seed(1)
su, se = 0.05, 0.5
qreal = su/se
e = np.random.normal(0, se, n)
u = np.random.normal(0, su, n)
z, wreal, const = 1, 0.86, 0.6
y = np.ones(n)
alpha = np.ones(n)
y[0], alpha[0] = const+e[0], const+u[0]
for t in range(1, n):
    y[t] = z*alpha[t-1]+e[t]
    alpha[t] = const+wreal*alpha[t-1]+u[t]
# standard Kalman filter approach
a, p, k, v = [np.ones(n) for _ in range(4)]
a[0], p[0] = 0, 10


def fu(mypa):
    w, se, su, co = abs(mypa[0]), abs(mypa[1]), abs(mypa[2]), abs(mypa[3])
    z, likelihood = 1, 0
    for t in range(1, n):
        k[t] = (z*w*p[t-1])/(pow(z, 2)*p[t-1]+se)
        p[t] = pow(w, 2)*p[t-1]-w*z*k[t]*p[t-1]+su
        v[t] = y[t]-z*a[t-1]
        a[t] = co+w*a[t-1]+k[t]*v[t]
        likelihood += .5*math.log(2*math.pi)+.5*math.log(pow(z, 2)*p[t-1]+se)+.5*(pow(v[t], 2)/(pow(z, 2)*p[t-1]+se))
    return likelihood

# likelihood<-likelihood+.5*log(2*pi)+.5*log(z^2*p[t-1]+se)+.5*(v[t]^2/(z^2*p[t-1]+se))
results = minimize(fu, [.85, .5, .3, .3], method='CG')
print("The results of the standard KF approach", '\n', np.round(np.abs(results.x), 4))
print("The true parameters", '\n', [wreal, se, su, const])

time = np.linspace(0, n-1, n)
plt.figure(figsize=[10, 5])
plt.plot(time, y)
plt.xlabel('index')
plt.ylabel('y')
plt.show()
