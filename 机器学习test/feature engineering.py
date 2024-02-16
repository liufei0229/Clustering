#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np #数维度数组与矩阵运算
import xlrd #读取excel
import pandas as pd #分析结构化数据
from matplotlib import pyplot as plt #可视化

io = r'E:\Python\HR模拟_1.xlsx'
df = pd.read_excel(io, sheet_name = "Sheet1")
df.head(10)


# In[20]:


#进行独热编码
dummies = pd.get_dummies(df['OnehotCoding'],prefix='独热编码')
dummies.head()


# In[21]:


#替换列值
df.Unit1.replace(["A","B","C"],["100","200","300"],inplace=True)
df.head()


# In[22]:


df.Unit1.replace(to_replace=dict(D=0),inplace=True)
df.head(10)


# In[23]:


def dataprocess(data):
    for col in ('feature1','feature2','feature3'):
        data[col] = np.where(data[col]==0,10000,5)
    return data
df=dataprocess(df)
df.head()


# In[ ]:




