import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("../../data/nao.csv", index_col=0)

data.head()
plt.figure(figsize=(18, 5), dpi=300)
data['nao'].plot()
data['diff'] = data['nao'].diff(1)
data.plot(subplots=True, figsize=(15, 5))
data.plot(subplots=True, figsize=(15, 5))
plt.show()


