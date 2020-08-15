import numpy as np


def random_walk(x0, sigma, const, n):
    a = np.random.normal(0, sigma, n)
    xt = np.zeros(n)
    xt[0] = x0
    for i in range(1, n):
        xt[i] = xt[i-1] + a[i] + const
    return xt
