import numpy as np


def hpfilter(xt, lamb=10):
    n = len(xt)
    d1 = d_matrix(n)
    d2 = d_matrix(n-1)
    d = d2 @ d1
    zt = np.linalg.inv((np.eye(n) + lamb * d.T @ d)) @ xt
    return zt


def d_matrix(n):
    d = np.zeros((n-1, n))
    d[:, 1:] = np.eye(n-1)
    d[:, :-1] -= np.eye(n-1)
    return d
