import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math


url = 'http://www.pangoo.it/coronavirus/?t=region&r=Lombardia&data=y#table'
data = pd.read_html(url)[0]
# data = pd.read_csv('cov19_data.csv')
count = data['Totale casi'][:-1]
n = len(count)
temp = np.array([])
for i in range(n):
    temp = np.append(temp, int(count[i]))
y = np.diff(temp)

time = np.linspace(1, n-1, n-1)
plt.figure(figsize=[10, 5])
plt.plot(time, y)
plt.xlabel('Index')
plt.ylabel('y')
plt.title('New Covid19 cases in Italy')
plt.show()


obs = len(y)-5
x = y[:obs]
a, p, k, v = [np.ones(n) for _ in range(4)]
a[0], p[0], v[0] = x[1], 10000, 0


def funcTheta(parameters):
    q, co = abs(parameters[0]), abs(parameters[1])
    z = w = 1
    likelihood = sigmae = 0
    for t in range(1, obs):
        k[t] = (z*w*p[t-1])/(pow(z, 2)*p[t-1]+1)
        p[t] = pow(w, 2)*p[t-1]-w*z*k[t]*p[t-1]+q
        v[t] = x[t]-z*a[t-1]
        a[t] = co+w*a[t-1]+k[t]*v[t]
        sigmae += (pow(v[t], 2)/(pow(z, 2)*p[t-1]+1))
        likelihood += .5*math.log(2*math.pi)+.5+.5*math.log(pow(z, 2)*p[t-1]+1)
    return likelihood+.5*n*math.log(sigmae/n)


results = minimize(funcTheta, [.6, .2])
z = w = 1
q, co, sigmae = results.x[0], results.x[1], 0

for t in range(1, obs):
    k[t] = (z*w*p[t-1])/(pow(z, 2)*p[t-1]+1)
    p[t] = pow(w, 2)*p[t-1]-w*z*k[t]*p[t-1]+q
    v[t] = x[t]-z*a[t-1]
    a[t] = co+w*a[t-1]+k[t]*v[t]
    sigmae += pow(v[t], 2)/(pow(z, 2)*p[t-1]+1)

#This is the drift parameter
print('co = ', np.round(co, 4))
#This is the variance of e
print('sigmae = ', np.round(sigmae/(n-1), 4))
#This is the variance of u
print('sigmau = ', np.round(q*(sigmae/(n-1)), 4))

t = 5
MyForecasts = np.ones(t)
# This is my one-step ahead for x:
MyForecasts[0] = a[obs-1]
MyForecasts[1] = co+MyForecasts[0]
MyForecasts[2] = co+MyForecasts[1]
MyForecasts[3] = co+MyForecasts[2]
MyForecasts[4] = co+MyForecasts[3]

time = np.linspace(0, t-1, t)
plt.figure()
plt.plot(time, y[len(y)-t:len(y)], time, MyForecasts, '--r')
plt.ylabel('y and forecast')
plt.title('black is y_t, red is the forecasts')
plt.show()

MASE = np.mean(np.abs(y[(len(y)-5):len(y)]-MyForecasts)/np.mean(np.abs(np.diff(x))))
print('MASE = ', np.round(MASE, 3))

v = np.array([3, 1, 4, 8, 2])
print(np.diff(v))

MAPE = np.mean(200*np.abs(y[(len(y)-5):len(y)]-MyForecasts)/(np.abs(MyForecasts)+np.abs(y[(len(y)-5):len(y)])))
print('MAPE = ', np.round(MAPE, 3))


a = 0*np.ones(obs)
v = 0*np.ones(obs)
a[0] = x[0]


def logLikConc(myparam):
    gamma = abs(myparam)
    w, z, co = 1, 1, 0
    for t in range(1, obs):
        v[t] = x[t]-z*a[t-1]
        a[t] = co+w*a[t-1]+gamma*v[t]
    temp = np.sum(np.power(v[1:obs], 2))
    return temp


myresults = minimize(logLikConc, 0.1, bounds=[[0, 1]])
w = z = 1
a = 0*np.ones(obs)
v = 0*np.ones(obs)
a[0] = x[0]
gamma = myresults.x[0]
for t in range(1, obs):
    v[t] = x[t] - z * a[t - 1]
    a[t] = co + w * a[t - 1] + gamma * v[t]

t = 5
LLForecasts = np.ones(t)
# This is my one-step ahead for x:
LLForecasts[0] = a[obs-1]
LLForecasts[1] = LLForecasts[0]
LLForecasts[2] = LLForecasts[1]
LLForecasts[3] = LLForecasts[2]
LLForecasts[4] = LLForecasts[3]

time = np.linspace(0, t-1, t)
plt.figure()
plt.plot(time, y[len(y)-t:len(y)], time, LLForecasts, '--r')
plt.ylabel('y and forecast')
plt.title('black is y_t, red is the forecasts')
plt.show()

MASETHETA = np.mean(np.abs(y[(len(y)-5):len(y)]-MyForecasts)/np.mean(np.abs(np.diff(x))))
MASELL = np.mean(np.abs(y[(len(y)-5):len(y)]-LLForecasts)/np.mean(np.abs(np.diff(x))))
print('MASETHETA = ', np.round(MASETHETA, 3), '\n', 'MASELL = ', np.round(MASELL, 3))


MAPETHETA = np.mean(200*np.abs(y[(len(y)-5):len(y)]-MyForecasts)/(np.abs(MyForecasts)+np.abs(y[(len(y)-5):len(y)])))
MAPELL = np.mean(200*np.abs(y[(len(y)-5):len(y)]-LLForecasts)/(np.abs(LLForecasts)+np.abs(y[(len(y)-5):len(y)])))
print('MAPETheta = ', np.round(MAPETHETA, 3), '\n', 'MAPELL = ', np.round(MAPELL, 3))
