#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%

#index_col  指定那一列为索引 默认就是系统自动给定索引 0-100
#sep = ","  按着，分隔符进行读取
data = pd.read_csv("../../data/gdp.csv", index_col=0)
 # 将字符串索引转换成时间索引
ts = data['index']  # 生成pd.Series对象
# 查看数据格式
ts.head()


#%%

data.tail()

#%%

plt.figure(figsize=(15, 5), dpi=300)
data['index'].plot(grid=True)

#%%

#平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF
print(u'原始序列的ADF检验结果为：', ADF(data['index']))

#%%

#进行一阶差分
data['diff_01'] = data['index'].diff(1)
#调整图片的大小和分辨率
plt.figure(figsize=(15,5),dpi=300)
#绘制差分后的时序图
data['diff_01'].plot(grid=True)

#%%

#差分后 删除缺失值
D_data = data.dropna()
#差分后平稳性检测
print(u'原始序列的ADF检验结果为：', ADF(D_data['diff_01']))

#%%

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#%%

fig = plt.figure(figsize=(15,5),dpi=300)
ax1 = fig.add_subplot(211)
fig = plot_acf(data['index'],lags=200,ax=ax1)
fig.tight_layout()

ax2 = fig.add_subplot(212)
fig = plot_pacf(data['index'],lags=200,ax=ax2)
fig.tight_layout()

#%%

# ARIMA 模型
# AR模型看pacf p 阶截尾  p=2
# MA模型看ACF q阶截尾    q = 2
#  d = 1
# 截尾 ：落在置信区间内

#%%



#%%

#白噪声检测
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data['diff_01'], lags=1)) #返回统计量和p值

#%%

#一阶差分后的时间序列是平稳非白噪声的时间序列数据

#%%

from statsmodels.tsa.arima_model import ARIMA

#%%

data = data.dropna()

#%%

ts_log = np.log(ts)

#%%


from statsmodels.tsa.arima_model import ARMA

model = ARMA(data['diff_01'], order=(2,2)).fit()

print(model.bic, model.aic, model.hqic)
#aic  bic hqic越小越好

#%%
