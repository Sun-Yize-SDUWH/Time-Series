import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


np.random.seed(1132)
n = 100
e = np.random.normal(0, 0.4, n)
y = np.empty(n)
alpha = np.empty(n)
s, co = 4, 0.3
sfactor = 10*np.random.uniform(0, 1, s)

y[0] = sfactor[0] + e[0]
y[1] = sfactor[1] + e[1]
y[2] = sfactor[2] + e[2]
y[3] = sfactor[3] + e[3]
alpha[0] = sfactor[0] + .2 * e[0]
alpha[1] = sfactor[1] + .2 * e[1]
alpha[2] = sfactor[2] + .2 * e[2]
alpha[3] = sfactor[3] + .2 * e[3]

for t in range(4, n):
    alpha[t] = co+alpha[t-s]+0.3*e[t]
    y[t] = alpha[t-s]+e[t]

time = np.linspace(0, n-1, n)
plt.figure(figsize=[10, 5])
plt.plot(time, y, time, alpha, '--')
plt.xlabel('index')
plt.ylabel('y')
plt.title('y and alpha')
plt.show()

s = 4
state = 0*np.ones(n)
e = 0*np.ones(n)
state[:s] = y[:s]


def logLikConc(myparam):
    gamma, co = abs(myparam[0]), abs(myparam[1])
    for t in range(s, n):
        e[t] = y[t]-state[t-s]
        state[t] = co+state[t-s]+gamma*e[t]
    temp = np.sum(np.power(e[1:], 2))/(n-1)
    return temp


myresults = minimize(logLikConc, [.2, .2])
print(myresults)
