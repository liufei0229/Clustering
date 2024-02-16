#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #数维度数组与矩阵运算
import xlrd #读取excel
import pandas as pd #分析结构化数据
from matplotlib import pyplot as plt #可视化

io = r'E:\Python\HR模拟_1.xlsx'
df = pd.read_excel(io, sheet_name = "Sheet1")
df.head(10)


# In[6]:


#进行独热编码
dummies = pd.get_dummies(df['OnehotCoding'],prefix='OnehotCoding')
dummies.head()


# In[7]:


new_df = df.join(dummies)
new_df.head()


# In[8]:


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

# In[9]:


from sklearn import preprocessing
#Z-Score标准化
#建立StandardScaler对象
zscore = preprocessing.StandardScaler()
# 标准化处理
data_zs_array = zscore.fit_transform(st_data)
df_data_zs=pd.DataFrame(data_zs_array,columns=["feature1_zs","feature2_zs","feature3_zs","feature4_zs","feature5_zs"])
df_data_zs.head()


# In[10]:


new_2df=new_df.join(df_data_zs)
new_2df.head()


# In[24]:



new_2df["Age"].describe()


# In[29]:


#绘制直方图
fig=plt.figure(figsize=(12,8))
p1=fig.add_subplot(221)
plt.hist(new_2df["Age"],bins=20,rwidth=0.9)
plt.xlabel('Age')
plt.xlim((18,66))
plt.ylabel('Counts')
plt.title('Age histogram')


p2=fig.add_subplot(222)
plt.hist(new_2df["Graduate"],bins=15,rwidth=0.9,color="green")
plt.xlabel('Graduate')
plt.xlim((0,8))
plt.ylabel('Counts')
plt.title('Graduate')


# In[36]:


#绘制线箱图

new_2df.boxplot(column='Age', by='Gender',showmeans=True)


# In[40]:


#相关系数矩阵
df_corr = new_2df.corr()

# 可视化
import seaborn
plt.figure(figsize=(20, 20))
seaborn.heatmap(df_corr, center=0, annot=True)
plt.show()


# In[48]:


#协方差矩阵

covariance_matrix=new_2df.cov()
plt.figure(figsize=(25, 25))

seaborn.heatmap(covariance_matrix, center=0, annot=True)

plt.show()


# In[ ]:




