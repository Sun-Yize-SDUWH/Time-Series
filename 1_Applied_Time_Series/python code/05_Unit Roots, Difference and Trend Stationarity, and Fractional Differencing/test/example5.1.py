import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import unit_root


# read the data
data = pd.read_csv('../../data/interest_rates.csv')
t = pd.date_range('1975-01-01', periods=len(data))
xt = np.array(data.iloc[:, 1])

tau = unit_root.unit_root([1.193, -0.224], xt)
print('tau = ', tau)

# plot
plt.figure()
plt.plot(t, xt)
plt.title('Observed')
plt.show()
