import numpy as np


def first_difference(xt):
    y = np.array([])
    for i in range(len(xt)-1):
        y = np.append(y, xt[i+1] - xt[i])
    return y


def autocorrelation_function(k, xt):
    mean = np.mean(xt)
    var = np.var(xt)
    temp = 0
    for i in range(k, len(xt)):
        temp += (xt[i] - mean)*(xt[i-k] - mean)
    r = temp/(len(xt)*var)
    return r


def moving_average(phi, sigma, const, n):
    k = len(phi)
    xt = np.zeros(n)
    a = np.random.normal(0, sigma, n)
    for i in range(k):
        xt[i] = a[i]
    for i in range(1, n):
        for t in range(k):
            if t == 0:
                temp = a[i]
            else:
                temp -= phi[t] * a[i - t]
        xt[i] = temp + const
    return xt
