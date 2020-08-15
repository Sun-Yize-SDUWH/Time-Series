import numpy as np
import pandas as pd
from scipy.optimize import minimize
import math


def ForecastARkf(y, h):
    n = len(y)
    a, p, k, v = [np.ones(n) for _ in range(4)]
    a[0], p[0] = y[0], 10000

    def fu(mypa):
        q, co, w = abs(mypa[0]), abs(mypa[1]), 1-math.exp(-abs(mypa[2]))
        z = 1
        likelihood = sigmae = 0
        for t in range(1, n):
            k[t] = (z * w * p[t - 1]) / (pow(z, 2) * p[t - 1] + 1)
            p[t] = pow(w, 2) * p[t - 1] - w * z * k[t] * p[t - 1] + q
            v[t] = y[t] - z * a[t - 1]
            a[t] = co + w * a[t - 1] + k[t] * v[t]
            sigmae += (pow(v[t], 2) / (pow(z, 2) * p[t - 1] + 1))
            likelihood += .5 * math.log(2 * math.pi) + .5 + .5 * math.log(pow(z, 2) * p[t - 1] + 1)
        return likelihood + .5 * n * math.log(sigmae / n)

    results = minimize(fu, [.2, 1, 2])
    v[0], z = 0, 1
    q, co, w, sigmae = abs(results.x[0]), results.x[1], 1-math.exp(-abs(results.x[2])), 0
    for t in range(1, n):
        k[t] = (z * w * p[t - 1]) / (pow(z, 2) * p[t - 1] + 1)
        p[t] = pow(w, 2) * p[t - 1] - w * z * k[t] * p[t - 1] + q
        v[t] = y[t] - z * a[t - 1]
        a[t] = co + w * a[t - 1] + k[t] * v[t]
        sigmae = sigmae + pow(v[t], 2) / (pow(z, 2) * p[t - 1] + 1)
    Forec = np.array([a[len(y)-1]])
    for i in range(1, h):
        Forec = np.append(Forec, co+w*Forec[i-1])
    return Forec


def ForecastAR(y, h):
    state = 0*np.ones(len(y))
    v = 0*np.ones(len(y))
    state[0] = y[0]

    def logLikConc(myparam):
        w, gamma, co = 1 - math.exp(-abs(myparam[0])), abs(myparam[1]), abs(myparam[2])
        for t in range(1, len(y)):
            v[t] = y[t]-state[t-1]
            state[t] = co+w*state[t-1]+gamma*v[t]
        temp = np.sum(np.power(v[1:len(y)], 2))
        return temp

    result = minimize(logLikConc, [2, .2, 1])
    w, gamma, co = 1 - math.exp(-abs(result.x[0])), abs(result.x[1]), abs(result.x[2])
    for t in range(1, len(y)):
        v[t] = y[t] - state[t - 1]
        state[t] = co + w * state[t - 1] + gamma * v[t]
    Forec = np.array([state[len(y) - 1]])
    for i in range(1, h):
        Forec = np.append(Forec, co + w * Forec[i - 1])
    return Forec


def ForecastTheta(y, h):
    state = 0 * np.ones(len(y))
    v = 0 * np.ones(len(y))
    state[0] = y[0]

    def logLikConc(myparam):
        w, gamma, co = 1, abs(myparam[0]), abs(myparam[1])
        for t in range(1, len(y)):
            v[t] = y[t]-state[t-1]
            state[t] = co+w*state[t-1]+gamma*v[t]
        temp = np.sum(np.power(v[1:len(y)], 2))
        return temp

    result = minimize(logLikConc, [.3, 1])
    w, gamma, co = 1, abs(result.x[0]), abs(result.x[1])
    for t in range(1, len(y)):
        v[t] = y[t] - state[t - 1]
        state[t] = co + w * state[t - 1] + gamma * v[t]
    Forec = np.array([state[len(y) - 1]])
    for i in range(1, h):
        Forec = np.append(Forec, co + w * Forec[i - 1])
    return Forec


def ForecastThetakf(y, h):
    n = len(y)
    a, p, k, v = [np.ones(n) for _ in range(4)]
    a[0], p[0], v[0] = y[0], 10000, 0

    def funcTheta(parameters):
        q, co = abs(parameters[0]), abs(parameters[1])
        z = w = 1
        likelihood = sigmae = 0
        for t in range(1, n):
            k[t] = (z * w * p[t - 1]) / (pow(z, 2) * p[t - 1] + 1)
            p[t] = pow(w, 2) * p[t - 1] - w * z * k[t] * p[t - 1] + q
            v[t] = y[t] - z * a[t - 1]
            a[t] = co + w * a[t - 1] + k[t] * v[t]
            sigmae += (pow(v[t], 2) / (pow(z, 2) * p[t - 1] + 1))
            likelihood += .5 * math.log(2 * math.pi) + .5 + .5 * math.log(pow(z, 2) * p[t - 1] + 1)
        return likelihood + .5 * n * math.log(sigmae / n)

    results = minimize(funcTheta, [.3, 1])
    q, co = abs(results.x[0]), abs(results.x[1])
    z = w = 1
    for t in range(1, n):
        k[t] = (z * w * p[t - 1]) / (pow(z, 2) * p[t - 1] + 1)
        p[t] = pow(w, 2) * p[t - 1] - w * z * k[t] * p[t - 1] + q
        v[t] = y[t] - z * a[t - 1]
        a[t] = co + w * a[t - 1] + k[t] * v[t]
    Forecast = np.array([a[len(y) - 1]])
    for i in range(1, h):
        Forecast = np.append(Forecast, co + w * Forecast[i - 1])
    return Forecast


def ForecastDamped(y, h):
    obs = len(y)
    damped = 0*np.ones([obs, 2])
    damped[0, 0] = y[0]
    damped[0, 1] = 0
    inn = 0*np.ones(obs)

    def fmsoe(param):
        k1, k2, k3 = abs(param[0]), abs(param[1]), abs(param[2])
        for t in range(1, obs):
            inn[t] = y[t]-damped[t-1, 0]-k3*damped[t-1, 1]
            damped[t, 0] = damped[t-1, 0]+k3*damped[t-1, 1]+k1*inn[t]
            damped[t, 1] = k3*damped[t-1, 1]+k2*inn[t]
        temp = np.sum(np.power(inn, 2)/obs)
        return temp

    result = minimize(fmsoe, np.random.uniform(0, 1, 3))
    k1, k2, k3 = abs(result.x[0]), abs(result.x[1]), abs(result.x[2])
    if k3 > 1:
        k3 = 1
    for t in range(1, obs):
        inn[t] = y[t] - damped[t - 1, 0] - k3 * damped[t - 1, 1]
        damped[t, 0] = damped[t - 1, 0] + k3 * damped[t - 1, 1] + k1 * inn[t]
        damped[t, 1] = k3 * damped[t - 1, 1] + k2 * inn[t]
    Forecast = np.array([damped[obs-1, 0]+k3 * damped[obs-1, 1]])
    for i in range(1, h):
        Forecast = np.append(Forecast, Forecast[i - 1] + damped[obs-1, 1] * np.power(k3, i))
    return Forecast


MM = pd.read_csv('../Mcomp_M3.csv', header=None)

# replic = len(MM)
replic = 50
steps = 6
Method1, Method2, Method3, Method4, Method5, Err1, Err2 = [0*np.ones([replic, steps]) for _ in range(7)]
Err3, Err4, Err5, sErr1, sErr2, sErr3, sErr4, sErr5 = [0*np.ones([replic, steps]) for _ in range(8)]
for g in range(replic):
    y = np.array(MM.iloc[g][:])
    y = np.array([x for x in y if not math.isnan(x)])[:-steps]
    # For the M4 competition use y<-MM[[g]][[2]]
    Method1[g][:] = ForecastAR(y, steps)
    Method2[g][:] = ForecastARkf(y, steps)
    Method3[g][:] = ForecastTheta(y, steps)
    Method4[g][:] = ForecastThetakf(y, steps)
    Method5[g][:] = ForecastDamped(y, steps)

    Err1[g][:] = np.array(MM.iloc[g][-steps:])-Method1[g][:]
    Err2[g][:] = np.array(MM.iloc[g][-steps:])-Method2[g][:]
    Err3[g][:] = np.array(MM.iloc[g][-steps:])-Method3[g][:]
    Err4[g][:] = np.array(MM.iloc[g][-steps:])-Method4[g][:]
    Err5[g][:] = np.array(MM.iloc[g][-steps:])-Method5[g][:]

    sErr1[g][:] = Err1[g][:] / np.mean(np.abs(np.diff(y)))
    sErr2[g][:] = Err2[g][:] / np.mean(np.abs(np.diff(y)))
    sErr3[g][:] = Err3[g][:] / np.mean(np.abs(np.diff(y)))
    sErr4[g][:] = Err4[g][:] / np.mean(np.abs(np.diff(y)))
    sErr5[g][:] = Err5[g][:] / np.mean(np.abs(np.diff(y)))
ResultsMAPE = 0*np.ones([steps, 18])

for s in range(steps):
    sMAPE = 0*np.ones([replic, 5])
    for i in range(replic):
        temp = np.array(MM.iloc[i][-steps:])
        sMAPE[i][0] = np.mean(200 * abs(Err1[i][0:s+1]) / (abs(Method1[i][:][0:s+1]) + abs(temp[0:s+1])))
        sMAPE[i][1] = np.mean(200 * abs(Err2[i][0:s+1]) / (abs(Method2[i][:][0:s+1]) + abs(temp[0:s+1])))
        sMAPE[i][2] = np.mean(200 * abs(Err3[i][0:s+1]) / (abs(Method3[i][:][0:s+1]) + abs(temp[0:s+1])))
        sMAPE[i][3] = np.mean(200 * abs(Err4[i][0:s+1]) / (abs(Method4[i][:][0:s+1]) + abs(temp[0:s+1])))
        sMAPE[i][4] = np.mean(200 * abs(Err5[i][0:s+1]) / (abs(Method5[i][:][0:s+1]) + abs(temp[0:s+1])))
    ResultsMAPE[s][0] = np.mean(sMAPE[:, 0])
    ResultsMAPE[s][1] = np.mean(sMAPE[:, 1])
    ResultsMAPE[s][2] = np.mean(sMAPE[:, 2])
    ResultsMAPE[s][3] = np.mean(sMAPE[:, 3])
    ResultsMAPE[s][4] = np.mean(sMAPE[:, 4])
    ResultsMAPE[s][5] = np.mean(sMAPE[:, 0]) / np.mean(sMAPE[:, 1])
    ResultsMAPE[s][6] = np.mean(sMAPE[:, 0]) / np.mean(sMAPE[:, 2])
    ResultsMAPE[s][7] = np.mean(sMAPE[:, 0]) / np.mean(sMAPE[:, 3])
    ResultsMAPE[s][8] = np.mean(sMAPE[:, 0]) / np.mean(sMAPE[:, 4])
    ResultsMAPE[s][9] = np.median(sMAPE[:, 0])
    ResultsMAPE[s][10] = np.median(sMAPE[:, 1])
    ResultsMAPE[s][11] = np.median(sMAPE[:, 2])
    ResultsMAPE[s][12] = np.median(sMAPE[:, 3])
    ResultsMAPE[s][13] = np.median(sMAPE[:, 4])
    ResultsMAPE[s][14] = np.median(sMAPE[:, 0]) / np.median(sMAPE[:, 1])
    ResultsMAPE[s][15] = np.median(sMAPE[:, 0]) / np.median(sMAPE[:, 2])
    ResultsMAPE[s][16] = np.median(sMAPE[:, 0]) / np.median(sMAPE[:, 3])
    ResultsMAPE[s][17] = np.median(sMAPE[:, 0]) / np.median(sMAPE[:, 4])

ResultsMASE = 0*np.ones([steps, 18])

for s in range(steps):
    sMASE = 0*np.ones([replic, 5])
    for i in range(replic):
        temp = np.array(MM.iloc[i][-steps:])
        sMASE[i][0] = np.mean(abs(sErr1[i][0:s+1]))
        sMASE[i][1] = np.mean(abs(sErr2[i][0:s+1]))
        sMASE[i][2] = np.mean(abs(sErr3[i][0:s+1]))
        sMASE[i][3] = np.mean(abs(sErr4[i][0:s+1]))
        sMASE[i][4] = np.mean(abs(sErr5[i][0:s+1]))
    ResultsMASE[s][0] = np.mean(sMASE[:, 0])
    ResultsMASE[s][1] = np.mean(sMASE[:, 1])
    ResultsMASE[s][2] = np.mean(sMASE[:, 2])
    ResultsMASE[s][3] = np.mean(sMASE[:, 3])
    ResultsMASE[s][4] = np.mean(sMASE[:, 4])
    ResultsMASE[s][5] = np.mean(sMASE[:, 0]) / np.mean(sMASE[:, 1])
    ResultsMASE[s][6] = np.mean(sMASE[:, 0]) / np.mean(sMASE[:, 2])
    ResultsMASE[s][7] = np.mean(sMASE[:, 0]) / np.mean(sMASE[:, 3])
    ResultsMASE[s][8] = np.mean(sMASE[:, 0]) / np.mean(sMASE[:, 4])
    ResultsMASE[s][9] = np.median(sMASE[:, 0])
    ResultsMASE[s][10] = np.median(sMASE[:, 1])
    ResultsMASE[s][11] = np.median(sMASE[:, 2])
    ResultsMASE[s][12] = np.median(sMASE[:, 3])
    ResultsMASE[s][13] = np.median(sMASE[:, 4])
    ResultsMASE[s][14] = np.median(sMASE[:, 0]) / np.median(sMASE[:, 1])
    ResultsMASE[s][15] = np.median(sMASE[:, 0]) / np.median(sMASE[:, 2])
    ResultsMASE[s][16] = np.median(sMASE[:, 0]) / np.median(sMASE[:, 3])
    ResultsMASE[s][17] = np.median(sMASE[:, 0]) / np.median(sMASE[:, 4])

print('ResultsMASE:', '\n', np.round(ResultsMASE, 3), '\n')
print('ResultsMASE col mean:', '\n', np.round(np.mean(ResultsMASE, axis=0), 3), '\n')
print('ResultsMAPE:', '\n', np.round(ResultsMAPE, 3), '\n')
print('ResultsMAPE col mean:', '\n', np.round(np.mean(ResultsMAPE, axis=0), 3), '\n')
