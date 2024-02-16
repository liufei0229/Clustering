#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np #数维度数组与矩阵运算
import xlrd #读取excel
import pandas as pd #分析结构化数据
from matplotlib import pyplot as plt #可视化

io = r'E:\Python\HR模拟_1.xlsx'
df = pd.read_excel(io, sheet_name = "Sheet1")
df.head(10)


# In[15]:


#进行独热编码
dummies = pd.get_dummies(df['OnehotCoding'],prefix='OnehotCoding')
dummies.head()


# In[18]:


new_df = df.join(dummies)
new_df.head()


# In[39]:


#创建需要标准化的数据表
print(len(df))
st_data=df.loc[:,["quantity1","quantity2","quantity3","quantity4","quantity5"]]
st_data.head()


# #Max-Min标准化
# #建立MinMaxScaler对象
# minmax = preprocessing.MinMaxScaler()
# data_minmax = minmax.fit_transform(data)
# 
# #MaxAbs标准化
# #建立MinMaxScaler对象
# maxabs = preprocessing.MaxAbsScaler()
# data_maxabs = maxabs.fit_transform(data)
# 
# #RobustScaler标准化
# #建立RobustScaler对象
# robust = preprocessing.RobustScaler()
# data_rob = robust.fit_transform(data)

# In[48]:


from sklearn import preprocessing
#Z-Score标准化
#建立StandardScaler对象
zscore = preprocessing.StandardScaler()
# 标准化处理
data_zs_array = zscore.fit_transform(st_data)
df_data_zs=pd.DataFrame(data_zs_array,columns=["feature1_zs","feature2_zs","feature3_zs","feature4_zs","feature5_zs"])
df_data_zs.head()


# In[49]:


new_df2=new_df.join(df_data_zs)
new_df2.head()


# In[ ]:




