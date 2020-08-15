import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import distributional


# read the data
data = pd.read_csv('../../data/rainfall.csv')
t = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d')
xt = data.iloc[:, 1]
fig1 = data.plot()
fig1.set_title('Observed')

# distributional transformation
transform1 = distributional.power_transformations(xt, 0.5)
transform2 = distributional.signed_power_transformation(xt, 0.5)

# plot
plt.figure()
plt.plot(t, transform2)
plt.title('Box–Cox transformed')
plt.show()
