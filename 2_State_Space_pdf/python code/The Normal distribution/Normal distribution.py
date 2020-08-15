import numpy as np
import matplotlib.pyplot as plt
import math


x = np.linspace(-3.8, 4.2, 800)
mu = 0.2
sigma = math.sqrt(1.5)
pdf = cdf = np.array([])
for i in range(len(x)):
    temp1 = 1/math.sqrt(2*math.pi*math.pow(sigma, 2))
    temp2 = math.exp(-.5*(math.pow(x[i]-mu, 2)/math.pow(sigma, 2)))
    pdf = np.append(pdf, temp1 * temp2)
    cdf = np.append(cdf, np.sum(pdf)/100)


plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x, pdf)
plt.xlabel('x')
plt.ylabel('pdf')
plt.title("Probability distribution function")
plt.subplot(2, 1, 2)
plt.plot(x, cdf)
plt.xlabel('x')
plt.ylabel('cdf')
plt.title("Cumulative distribution function")
plt.show()
