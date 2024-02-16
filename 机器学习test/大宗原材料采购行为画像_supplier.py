#!/usr/bin/env python
# coding: utf-8

# In[79]:


import numpy as np #数维度数组与矩阵运算
import xlrd #读取excel
import pandas as pd #分析结构化数据
from matplotlib import pyplot as plt #可视化


io = r'E:/大宗原材料采购行为预测/supplier_energy.xlsx.'
df = pd.read_excel(io, sheet_name = "Sheet1")
print(type(df))
df.head(10)


# In[80]:


df.describe()


# In[81]:


st_data=df.loc[:,["money","material_number"]]
st_data.head()


# In[82]:


from sklearn import preprocessing
#Z-Score标准化
#建立StandardScaler对象
zscore = preprocessing.StandardScaler()
# 标准化处理
data_zs_array = zscore.fit_transform(st_data)
df_data_zs=pd.DataFrame(data_zs_array,columns=["z_money","z_material_number"])
df_data_zs.head()


# In[83]:


from sklearn.cluster import KMeans
Inertias_list = []
score_list=[]
from sklearn.metrics import silhouette_samples,silhouette_score
for cluster in range(2,10):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster)
    kmeans.fit(df_data_zs)
    silhouette_avg= silhouette_score(df_data_zs,kmeans.labels_)
    Inertias_list.append(kmeans.inertia_)
    score_list.append(silhouette_avg)
    
    print("for cluster number is :",cluster,"\n",
          "The SSE is :",kmeans.inertia_,"\n",
          "The average silhouette_score is :",silhouette_avg,"\n","_______________________")
    
print(Inertias_list)
print(type(Inertias_list))


# In[84]:


# 绘制图形
Cluster=range(2,10)
frame = pd.DataFrame({'Cluster':Cluster, 'Inertia':Inertias_list})
from matplotlib import pyplot as plt #可视化
plt.figure(figsize=(8,4))
plt.plot(Cluster,Inertias_list,color="g",marker="o",linestyle="-.")
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
#添加数据标记
for x,y in zip(Cluster,Inertias_list):
    plt.text(x,y,'%.0f' %y,fontdict={'fontsize':10})


# In[85]:


from sklearn.cluster import KMeans
n_clusters=4
kmeans=KMeans(n_clusters=n_clusters)
kmeans.fit(df_data_zs)
centers = kmeans.cluster_centers_
centers


# In[86]:


arr=np.insert(centers,0,values=range(n_clusters),axis=1)
arr


# In[87]:


center_pd=pd.DataFrame(arr,columns=["cluster_No.","c_money","c_material_number"])
center_pd


# In[88]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(16,10))
x=df["money"]
y=df["material_number"]
y_predict=kmeans.predict(df_data_zs)

p1=fig.add_subplot(221)
plt.scatter(x,y,c="black")

p2=fig.add_subplot(222)
plt.scatter(x,y,c=y_predict)


# In[89]:


#返回每个分类的个数 np.array计数
kind,count=np.unique(kmeans.labels_,return_counts=True)
print(kind)
print(count)


# In[90]:


#打印出每个序号样本对应的类别分类， 0,1,2,3，和上面的中心点一一对应
km_lables=pd.DataFrame(kmeans.labels_,columns=["cluster_No."])
df_final=df.join(km_lables)
df_final.head(10)


# In[91]:


from sklearn.metrics import silhouette_samples,silhouette_score
#调用轮廓系数分数，silhouette_avg生成所有样本点轮廓系数的均值
#需要输入两个参数，特征矩阵X与聚类完毕的标签
silhouette_avg = silhouette_score(df_data_zs,kmeans.labels_)
#打印现有簇数量下，轮廓系数是多少
print("For n_clusters =", n_clusters,
      "The average silhouette_score is :", silhouette_avg)


# In[92]:


#调用silhouette_samples，返回每个样本点的轮廓系数，就是横坐标
sample_silhouette_values = silhouette_samples(df_data_zs, kmeans.labels_)
print(sample_silhouette_values)


# In[93]:


kmeans.inertia_


# In[94]:


#导出到excel
df_final.to_excel('clustering_supplierC10513.xlsx',index = False)


# In[ ]:




