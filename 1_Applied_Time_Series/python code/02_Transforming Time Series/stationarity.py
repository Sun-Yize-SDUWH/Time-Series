import numpy as np


def first_difference(xt):
    y = np.array([])
    for i in range(len(xt)-1):
        y = np.append(y, xt[i+1] - xt[i])
    return y


def second_differences(xt):
    y = np.array([])
    for i in range(len(xt) - 2):
        y = np.append(y, xt[i + 2] - 2 * xt[i+1] + xt[i])
    return y
