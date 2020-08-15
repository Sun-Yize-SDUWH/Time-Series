import numpy as np
import math


def power_transformations(xt, lam):
    if lam == 0:
        return np.log(xt)
    else:
        return (np.power(xt, lam)-1)/lam


def signed_power_transformation(xt, lam):
    if lam > 0:
        return (np.sign(xt) * np.abs(np.power(xt, lam) - 1)) / lam


def generalized_power(xt, lam):
    y = np.array([])
    for i in range(len(xt)):
        if xt[i] >= 0 and lam != 0:
            y = np.append(y, (np.power(xt[i] + 1, lam) - 1) / lam)
        elif xt[i] >= 0 and lam == 0:
            y = np.append(y, math.log(xt[i]+1))
        elif xt[i] < 0 and lam != 2:
            y = np.append(y, -(np.power(1-xt[i], 2-lam)-1)/(2-lam))
        else:
            y = np.append(y, -math.log(1-xt[i]))
    return y


def inverse_hyperbolic_sine(xt, lam):
    if lam > 0:
        return np.log((np.log(lam * xt + np.power(((np.power(lam, 2)) * np.power(xt, 2) + 1), 0.5))) / lam)
