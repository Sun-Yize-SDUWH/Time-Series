import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


mx = my = 0
varx = 0.5
vary = 0.6
covxy = -0.3
sigma = np.zeros([2, 2])
sigma[0][0] = varx
sigma[1][1] = vary
sigma[0][1] = sigma[1][0] = covxy
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
[X, Y] = np.meshgrid(x, y)


def f(x, y):
    temp1 = 1/(2*math.pi*np.sqrt(np.linalg.det(sigma)))
    temp2 = math.exp(-.5*((vary*math.pow(x-mx, 2)+(y-my)*(-2*(x-mx)*covxy+varx*(y-my)))/(varx*vary-2*covxy)))
    return temp1*temp2


z = np.empty([len(x), len(y)])
for i in range(len(x)):
    for j in range(len(y)):
        z[i][j] = f(x[i], y[j])

fig = plt.figure()
Axes3D(fig).plot_surface(X, Y, z, rstride=1, cstride=1, cmap='rainbow')
plt.figure()
plt.subplot(1, 2, 1)
plt.contourf(X, Y, z)
plt.xlabel('x')
plt.ylabel('y')
plt.subplot(1, 2, 2)
ctr = plt.contour(X, Y, z)
plt.clabel(ctr, fontsize=10, colors='k')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
