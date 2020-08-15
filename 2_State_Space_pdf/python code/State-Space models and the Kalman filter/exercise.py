import numpy as np
import matplotlib.pyplot as plt


def StateSpaceGen(param):
    sigmae, sigmau, z, w, const = param[0], param[1], param[2], param[3], param[4]
    n = 100
    e = np.random.normal(0, sigmae, n)
    u = np.random.normal(0, sigmau, n)
    y = np.ones(n)
    alpha = np.ones(n)
    y[0], alpha[0] = e[0], u[0]
    for t in range(1, n):
        y[t] = z*alpha[t-1]+e[t]
        alpha[t] = const+w*alpha[t-1]+u[t]
    return np.array([y, alpha])


def KF(param):
    sigmae, sigmau, z, w, const = param[0], param[1], param[2], param[3], param[4]
    y = param[5]
    n = len(y)
    a, p, k, v = [np.ones(n) for _ in range(4)]
    a[0], p[0] = y[0], 10000
    if w < 1:
        a[0] = 0
        p[0] = sigmau/(1-pow(w, 2))
    for t in range(1, n):
        k[t] = (z*w*p[t-1])/(pow(z, 2)*p[t-1]+sigmae)
        p[t] = pow(w, 2)*p[t-1]-w*z*k[t]*p[t-1]+sigmau
        v[t] = y[t]-z*a[t-1]
        a[t] = const+w*a[t-1]+k[t]*v[t]
    return np.array([a, v, k, p])


n = 100
np.random.seed(222)
time = np.linspace(0, n-1, n)
y1 = StateSpaceGen([.5, .1, 1, .8, .3])
y2 = KF([.5, .1, 1, .8, .3, y1[0]])
plt.figure(figsize=[10, 5])
plt.plot(time, y1[0], time, y1[1], '--', time, y2[0], '--')
plt.legend(['y', 'alpha', 'a'])
plt.ylabel('combine(y,alpha,a)')
plt.title('y,alpha,a')
plt.show()
