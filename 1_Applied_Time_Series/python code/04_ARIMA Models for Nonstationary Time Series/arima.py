import numpy as np


def autoregressive_integrated_moving_average(d, phi, theta, sigma, x0, const, n):
    k1 = len(phi)
    k2 = len(theta)
    xt = np.zeros(n)
    a = np.random.normal(0, sigma, n)
    for i in range(k1):
        xt[i] = x0[i]
    for i in range(k1, n):
        temp = 0
        for t in range(k1):
            temp += phi[t] * xt[i - t - 1]
        for t in range(k2):
            if t == 0:
                temp += a[i]
            else:
                temp -= theta[t] * a[i - t]
        xt[i] = temp + const
    for i in range(len(x0)):
        xt[i] = x0[i]
    xt = difference_calculate(xt, d)
    return xt


def difference_calculate(xt, d):
    if d == 0:
        return xt
    if d == 1:
        for i in range(1, len(xt)):
            xt[i] = xt[i-1] + xt[i]
        return xt
    if d == 2:
        for i in range(2, len(xt)):
            temp = xt[i] + 2 * xt[i-1] - xt[i-2]
            xt[i] = temp
        return xt
