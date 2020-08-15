import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


np.random.seed(1)
n = 100
x = np.random.uniform(1, 8, n)
y = -1*np.random.normal(0, 1, n) + 0.8*x
plt.figure()
plt.scatter(x, y)
plt.show()


def myfu(beta):
    b = beta
    e = y-b*x
    return np.sum(np.power(e, 2))


myrelation = minimize(myfu, 1, method='CG')
print(myrelation)
