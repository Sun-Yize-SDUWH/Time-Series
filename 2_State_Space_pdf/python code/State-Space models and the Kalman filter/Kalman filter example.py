import numpy as np
import matplotlib.pyplot as plt


n = 100
np.random.seed(1123)
e = np.random.normal(0, 0.8, n)
u = np.random.normal(0, 0.4, n)
y = np.empty(n)
alpha = np.empty(n)
y[0] = e[0]
alpha[0] = u[0]

for t in range(1, n):
    y[t] = alpha[t-1]+e[t]
    alpha[t] = 0.9*alpha[t-1]+u[t]

time = np.linspace(0, n-1, n)
plt.figure(figsize=[10, 5])
plt.plot(time, alpha)
plt.scatter(time, y, facecolors='none', edgecolors='b')
plt.show()


n = 100
sigmae, sigmau = .8, .4
w, z = 0.9, 1
a, p, k, v = [np.ones(n) for _ in range(4)]
a[0] = 0
p[0] = 2.11

for t in range(1, n):
    k[t] = (z*w*p[t-1])/(pow(z, 2)*p[t-1]+sigmae)
    p[t] = pow(w, 2)*p[t-1]-w*z*k[t]*p[t-1]+sigmau
    v[t] = y[t]-z*a[t-1]
    a[t] = w*a[t-1]+k[t]*v[t]

plt.figure(figsize=[10, 5])
plt.plot(time, y, time, alpha, '--', time, a, '--')
plt.legend(['y', 'alpha', 'a'])
plt.ylabel('combine(y,alpha,a)')
plt.title('y,alpha,a')
plt.show()
