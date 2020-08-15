import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


np.random.seed(213)
n = 100
e = np.random.normal(0, 0.6, n)
gamma = 0.3
y = np.empty(n)
alpha = np.empty(n)
y[0] = e[0]
alpha[0] = e[0]
for t in range(1, n):
    y[t] = alpha[t-1]+e[t]
    alpha[t] = alpha[t-1]+gamma*e[t]

time = np.linspace(0, n-1, n)
plt.figure(figsize=[10, 5])
plt.plot(time, y, time, alpha)
plt.xlabel('index')
plt.ylabel('y')
plt.title('y and alpha')
plt.show()

a = np.empty(n)
a[0] = y[0]
z = 1
e = np.empty(len(y))


def fu(mypa):
    gamma = abs(mypa)
    for t in range(1, n):
        e[t] = y[t]-z*a[t-1]
        a[t] = a[t-1]+gamma*e[t]
    temp = np.sum(np.power(e, 2))/n
    return temp

results = minimize(fu, [.2], bounds=[[0, 1]])
print('gamma = ', results.x[0])
