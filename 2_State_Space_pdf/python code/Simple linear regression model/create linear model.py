import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1)
x = np.random.chisquare(6, 40)
e = np.random.normal(0, 1, 40)
y = -1.3 + 0.8*x + e
pointsline = -1.3 + 0.8*np.arange(0, 20)

plt.close()
plt.figure()
plt.plot(x, y, '*', np.arange(0, 20), pointsline)
plt.title("What is the relation between x and y?")
plt.xlabel('x')
plt.ylabel('y')
plt.show()
