import numpy as np


def moving_average(xt, n):
    y = np.array([])
    if n % 2 == 1:
        for i in range(len(xt)-n+1):
            temp = 0
            for j in range(n):
                temp += xt[i+j]
            temp = temp / n
            y = np.append(y, temp)
        return y
    else:
        for i in range(len(xt)-n):
            temp = 0
            for j in range(n+1):
                if j == 0 or j == n:
                    temp += xt[i+j] / (2*n)
                else:
                    temp += xt[i+j] / n
            y = np.append(y, temp)
        return y
