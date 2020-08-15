import numpy as np


def unit_root(phi, xt):
    t = len(xt)
    k = len(phi)
    temp1 = temp2 = 0
    for i in range(1, t):
        temp1 += np.power(xt[i-1], 2)
        temp3 = xt[i]
        for j in range(k):
            temp3 -= phi[j]*xt[i-j-1]
        temp2 += np.power(temp3, 2)
    st2 = temp2 / (t-1)
    se = np.power((st2 / temp1), 0.5)
    tau = (np.sum(phi)-1) / se
    return tau
