import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math


np.random.seed(1224)
x = np.sin(np.arange(0.01, 1, 0.01))

b = np.random.uniform(0, 1, 1)
y = -b*x + 0.03*np.random.normal(0, 1, len(x))

plt.figure()
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


def myfu(beta):
    b = beta
    e = y - b*x
    return sum(abs(e))


myrelation = minimize(myfu, 1, method='CG')
print('b = ', b[0])
print('predict = ', abs(myrelation.x[0]), '\n')
print(myrelation)


def myfun(g):
    y = math.exp(math.pow(-.5*(g-4), 2)) / math.pow(2*math.pi, 2)
    return y


myrelation = minimize(myfun, 1, method='CG')
print(myrelation)


def myfun(myvalues):
    x = myvalues[0]
    y = myvalues[1]
    return math.pow(x-2, 2)+math.pow(4+y, 2)


myrelation = minimize(myfun, [1, 3])
print(myrelation)

from mpl_toolkits.mplot3d import Axes3D


x = np.linspace(-5, 10, 50)
y = np.linspace(-11, 2, 50)
[X, Y] = np.meshgrid(x, y)

def f(m, n):
    return np.power(-2+m, 2)+np.power(4+n, 2)


z = np.zeros([len(x), len(y)])
for i in range(len(x)):
    for j in range(len(y)):
        z[i][j] = f(x[i], y[j])

fig = plt.figure()
Axes3D(fig).plot_surface(X, Y, z, rstride=1, cstride=1, cmap='rainbow')
plt.figure()
plt.subplot(1, 2, 1)
plt.contourf(X, Y, z)
plt.subplot(1, 2, 2)
ctr = plt.contour(X, Y, z)
plt.clabel(ctr, fontsize=10, colors='k')
plt.show()


x = np.linspace(-5, 10, 50)
y = np.linspace(-11, 2, 50)


def f(x, y):
    return -math.pow(-2+x, 2)-math.pow(4+y, 2)


z = np.zeros([len(x), len(y)])
for i in range(len(x)):
    for j in range(len(y)):
        z[i][j] = f(x[i], y[j])

fig = plt.figure()
Axes3D(fig).plot_surface(X, Y, z, rstride=1, cstride=1, cmap='rainbow')
plt.show()
