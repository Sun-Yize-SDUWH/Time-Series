import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import math


def ForecastNN(y, h):
    n = len(y)
    k = 3
    data = np.ones([n-k, k+1])
    for i in range(n-k):
        data[i] = y[i:i+k+1]
    np.random.shuffle(data)
    xtrain = data[:, :k]
    # ytrain = data[:, k].astype('int')
    ytrain = data[:, k].astype('int')
    nn = RandomForestClassifier(n_estimators=10)
    nn.fit(xtrain, ytrain)
    Forec = np.ones(h)
    Forec[0] = nn.predict(np.array(y[-k:]).reshape(1, -1))
    for i in range(1, h):
        if i < k:
            Forec[i] = nn.predict(np.append(y[-k+i:], Forec[:i]).reshape(1, -1))
        else:
            Forec[i] = nn.predict(np.array(Forec[i-k:i]).reshape(1, -1))
    return Forec


data = pd.read_csv('../SP100list.csv')
# n = 301
# MM = np.array(data).transpose()
# MM = pd.DataFrame(MM)

n = len(data)//7
MM = pd.DataFrame(np.ones([101, n]))
for i in range(n):
    MM[:][i] = np.array(data.iloc[i*7][:])

print(MM.iloc[2, :-1])
data = ForecastNN(MM.iloc[2, :-1], 6)
print(data, MM.iloc[2, -1])
# print(data[0])


# def polyfit(y, h):
#     x = np.linspace(0, len(y)+h-1, len(y)+h)
#     p = np.poly1d(np.polyfit(x[:-h], y, 3))
#     Forec = p(x[-h:])
#     return Forec
#
# data = pd.read_csv('../SP100list.csv')
# # n = 301
# # MM = np.array(data).transpose()
# # MM = pd.DataFrame(MM)
#
# n = len(data)//7
# MM = pd.DataFrame(np.ones([101, n]))
# for i in range(n):
#     MM[:][i] = np.array(data.iloc[i*7][:])
#
# h = 6
# y = MM.iloc[6][:]


# predict = polyfit(y[:-h], h)
# print(predict)




# x = np.linspace(0, h-1, h)
# plt.figure()
# plt.plot(np.linspace(0, len(data)-1, len(data)), data)
# # plt.figure()
# # plt.plot(np.linspace(0, h-1, h), y[-h:], x, predict)
# plt.show()
