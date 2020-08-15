import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import sys
sys.path.append("..")


data = pd.read_csv('../../data/interest_rates.csv')
temp = data.iloc[:, 0]
t = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d')
r20 = np.array(data.iloc[:, 1])
rs = np.array(data.iloc[:, 2])

k1 = st.linregress(rs, r20)
k2 = st.linregress(r20, rs)

print(k1)
# e1 = r20 - b1 - k1*rs
# e2 = rs - b2 - k2*r20
# print(e1, e2)

plt.flag()
plt.plot(r20, rs, '*')
plt.show()
