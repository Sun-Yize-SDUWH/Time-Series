#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%%

data = pd.read_csv("../../data/wine_spirits.csv",index_col=0)

#%%

data.head()

#%%

plt.figure(figsize=(15, 5), dpi=300)
data.plot(grid=True)
plt.show()

#%%
