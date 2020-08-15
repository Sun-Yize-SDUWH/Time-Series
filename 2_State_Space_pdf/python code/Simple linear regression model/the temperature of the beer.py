import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


n = 200
np.random.seed(6)
gender = np.round(1.1*np.random.uniform(0, 1, n), 0)
temperature = -2 + np.arange(0, 9.99, .05)
ef = -.4 * np.random.normal(0, 1, n)
em = -.8 * np.random.normal(0, 1, n)
appreciation = np.zeros(n)

for i in range(n):
    if gender[i] == 0:
        appreciation[i] = -7-0.3*temperature[i]+ef[i]
    else:
        appreciation[i] = -10-0.7*temperature[i]+em[i]

plt.figure(figsize=[10, 4])
plt.subplot(1, 2, 1)
plt.scatter(temperature, appreciation)
plt.xlabel('temperature')
plt.ylabel('appreciation')
plt.subplot(1, 2, 2)
plt.scatter(temperature, appreciation, cmap='rainbow', c=gender)
plt.xlabel('temperature')
plt.ylabel('appreciation')
plt.show()

data = pd.DataFrame(np.array([appreciation, temperature]).transpose())
OurRegression = data.describe()
print(OurRegression)
