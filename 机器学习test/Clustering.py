#!/usr/bin/env python
# coding: utf-8

# # Kmeans

# In[8]:


import numpy as np #数维度数组与矩阵运算
import xlrd #读取excel
import pandas as pd #分析结构化数据
from matplotlib import pyplot as plt #可视化


io = r'E:/Python/机器学习test/game_ori.xlsx.'
df = pd.read_excel(io, sheet_name = "Sheet1")
print(type(df))
df.head(10)


# In[4]:


df.describe()


# In[9]:


st_data=df.loc[:,["last_login_to_now","duration","level_end","os_corr" ,"active_days" ,"avg_session_cnt" ,"is_payer"]]
del st_data["is_payer"]
st_data.head()


# In[10]:


from sklearn import preprocessing
#Z-Score标准化
#建立StandardScaler对象
zscore = preprocessing.StandardScaler()
# 标准化处理
data_zs_array = zscore.fit_transform(st_data)
df_data_zs=pd.DataFrame(data_zs_array,columns=["last_login_to_now1","duration1","level_end1","os_corr1" ,"active_days1" ,"avg_session_cnt1"])
df_data_zs.head()


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
X=df_data_zs.loc[:,["last_login_to_now1","active_days1"]]
#可视化
plt.scatter(X["last_login_to_now1"],X["active_days1"],c='black')
plt.xlabel('active days')
plt.ylabel('last login to now')
plt.show()


# In[12]:


#簇的个数
K=5
# 随机选择观察值作为簇心
Centroids = (X.sample(n=K))
plt.scatter(X["last_login_to_now1"],X["active_days1"],c='black')
plt.scatter(Centroids["last_login_to_now1"],Centroids["last_login_to_now1"],c='red')
plt.xlabel('active days')
plt.ylabel('last login to now')
plt.show()


# In[13]:


print(df_data_zs.isnull().any())


# In[14]:


from sklearn.cluster import KMeans
n_clusters=4
kmeans=KMeans(n_clusters=n_clusters)
kmeans.fit(df_data_zs)
centers = kmeans.cluster_centers_
centers

参数 	                   说明
n-cluster 	             分类簇的数量

max_iter 	             最大的迭代次数

n_init 	                 算法的运行次数

init 	                 接收待定的string。kmeans++表示该初始化策略选择的初始均值向量之间都距离比较远，它的效果较好；
                         random表示从数据中随机选择K个样本最为初始均值向量；或者提供一个数组，数组的形状为（n_cluster,n_features），
                         该数组作为初始均值向量。
                         
precompute_distance 	 接收Boolean或者auto。表示是否提前计算好样本之间的距离，auto表示如果nsamples*n>12 million，则不提前计算。

tol                      接收float，表示算法收敛的阈值。

N_jobs 	                 表示任务使用CPU数量。

random_state 	         表示随机数生成器的种子。

verbose 	             0表示不输出日志信息；1表示每隔一段时间打印一次日志信息。如果大于1，打印次数频繁。
# In[15]:


arr=np.insert(centers,0,values=range(n_clusters),axis=1)
center_pd=pd.DataFrame(arr,columns=["cluster_No.","last_login_to_now","duration","level_end","os_corr" ,"active_days" ,"avg_session_cnt" ])
center_pd


# In[16]:


#返回每个分类的个数 np.array计数
kind,count=np.unique(kmeans.labels_,return_counts=True)
print(kind)
print(count)


# In[17]:


#打印出每个序号样本对应的类别分类， 0,1,2,3，和上面的中心点一一对应
km_lables=pd.DataFrame(kmeans.labels_,columns=["cluster_No."])
df_final=df.join(km_lables)
df_final.head(100)


# In[18]:


#导出到excel
df_final.to_excel('clustering sample.xlsx',index = False)


# In[19]:


from sklearn.metrics import silhouette_samples,silhouette_score
#调用轮廓系数分数，silhouette_avg生成所有样本点轮廓系数的均值
#需要输入两个参数，特征矩阵X与聚类完毕的标签
silhouette_avg = silhouette_score(df_data_zs,kmeans.labels_)
#打印现有簇数量下，轮廓系数是多少
print("For n_clusters =", n_clusters,
      "The average silhouette_score is :", silhouette_avg)


# In[68]:


#调用silhouette_samples，返回每个样本点的轮廓系数，就是横坐标
sample_silhouette_values = silhouette_samples(df_data_zs, kmeans.labels_)
print(sample_silhouette_values)


# In[20]:


#碎石图的inertia值
kmeans.inertia_


# In[33]:


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


# In[34]:


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


# In[29]:


Cluster=range(2,25)
frame1 = pd.DataFrame({'Cluster':Cluster, 'average silhouette_score ':score_list})
frame1.head(10)


# In[ ]:





# # DBSCAN

# In[42]:


from sklearn.cluster import DBSCAN
from sklearn import metrics
# 计算DBSCAN eps是最小圆距离，min_samples是每个簇最小点数
db = DBSCAN(eps=0.7, min_samples=10).fit(df_data_zs)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
db_labels = db.labels_
db_labels


# In[47]:


# 聚类的结果
n_clusters_ = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise_ = list(db_labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

1）eps： DBSCAN算法参数，即我们的ϵ-邻域的距离阈值，和样本距离超过ϵ的样本点不在ϵ-邻域内。默认值是0.5.一般需要通过在多组值里面选择一个合适的阈值。eps过大，则更多的点会落在核心对象的ϵ

-邻域，此时我们的类别数可能会减少， 本来不应该是一类的样本也会被划为一类。反之则类别数可能会增大，本来是一类的样本却被划分开。

2）min_samples： DBSCAN算法参数，即样本点要成为核心对象所需要的ϵ

-邻域的样本数阈值。默认值是5. 一般需要通过在多组值里面选择一个合适的阈值。通常和eps一起调参。在eps一定的情况下，min_samples过大，则核心对象会过少，此时簇内部分本来是一类的样本可能会被标为噪音点，类别数也会变多。反之min_samples过小的话，则会产生大量的核心对象，可能会导致类别数过少。

3）metric：最近邻距离度量参数。可以使用的距离度量较多，一般来说DBSCAN使用默认的欧式距离（即p=2的闵可夫斯基距离）就可以满足我们的需求。可以使用的距离度量参数有：

　　　　a) 欧式距离 “euclidean”: ∑i=1n(xi−yi)2−−−−−−−−−−√

　　　　b) 曼哈顿距离 “manhattan”： ∑i=1n|xi−yi|

　　　　c) 切比雪夫距离“chebyshev”: max|xi−yi|(i=1,2,...n)

　　　　d) 闵可夫斯基距离 “minkowski”: ∑i=1n(|xi−yi|)p−−−−−−−−−−−√p

        p=1为曼哈顿距离， p=2为欧式距离。

　　　　e) 带权重闵可夫斯基距离 “wminkowski”: ∑i=1n(w∗|xi−yi|)p−−−−−−−−−−−−−−√p

其中w为特征权重

　　　　f) 标准化欧式距离 “seuclidean”: 即对于各特征维度做了归一化以后的欧式距离。此时各样本特征维度的均值为0，方差为1.

　　　　g) 马氏距离“mahalanobis”：(x−y)TS−1(x−y)−−−−−−−−−−−−−−−√
其中，S−1

为样本协方差矩阵的逆矩阵。当样本分布独立时， S为单位矩阵，此时马氏距离等同于欧式距离。

　　还有一些其他不是实数的距离度量，一般在DBSCAN算法用不上，这里也就不列了。

4）algorithm：最近邻搜索算法参数，算法一共有三种，第一种是蛮力实现，第二种是KD树实现，第三种是球树实现。这三种方法在K近邻法(KNN)原理小结中都有讲述，如果不熟悉可以去复习下。对于这个参数，一共有4种可选输入，‘brute’对应第一种蛮力实现，‘kd_tree’对应第二种KD树实现，‘ball_tree’对应第三种的球树实现， ‘auto’则会在上面三种算法中做权衡，选择一个拟合最好的最优算法。需要注意的是，如果输入样本特征是稀疏的时候，无论我们选择哪种算法，最后scikit-learn都会去用蛮力实现‘brute’。个人的经验，一般情况使用默认的 ‘auto’就够了。 如果数据量很大或者特征也很多，用"auto"建树时间可能会很长，效率不高，建议选择KD树实现‘kd_tree’，此时如果发现‘kd_tree’速度比较慢或者已经知道样本分布不是很均匀时，可以尝试用‘ball_tree’。而如果输入样本是稀疏的，无论你选择哪个算法最后实际运行的都是‘brute’。

5）leaf_size：最近邻搜索算法参数，为使用KD树或者球树时， 停止建子树的叶子节点数量的阈值。这个值越小，则生成的KD树或者球树就越大，层数越深，建树时间越长，反之，则生成的KD树或者球树会小，层数较浅，建树时间较短。默认是30. 因为这个值一般只影响算法的运行速度和使用内存大小，因此一般情况下可以不管它。

6） p: 最近邻距离度量参数。只用于闵可夫斯基距离和带权重闵可夫斯基距离中p值的选择，p=1为曼哈顿距离， p=2为欧式距离。如果使用默认的欧式距离不需要管这个参数。
# In[48]:


from sklearn.metrics import silhouette_samples,silhouette_score
#调用轮廓系数分数，silhouette_avg生成所有样本点轮廓系数的均值
#需要输入两个参数，特征矩阵X与聚类完毕的标签
silhouette_avg = silhouette_score(df_data_zs,db_labels)
#打印现有簇数量下，轮廓系数是多少
print("For n_clusters =", n_clusters_,
      "The average silhouette_score is :", silhouette_avg)


# In[72]:


Inertias_list= []
score_list=[]
from sklearn.metrics import silhouette_samples,silhouette_score
minpoints=5
for eps10 in range(1,20):
    db = DBSCAN(eps=eps10/10, min_samples=minpoints).fit(df_data_zs)
    n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    silhouette_avg= silhouette_score(df_data_zs,db.labels_)
    n_noise_ = list(db.labels_).count(-1)
    Inertias_list.append(kmeans.inertia_)
    score_list.append(silhouette_avg)
    
    print("for epsilon is :",eps10/10,"\n",
          "and for minpoints is:", minpoints, "\n",
          "the cluster number is", n_clusters_,"\n",
          "noise points are",n_noise_,"\n"
          "The average silhouette_score is :",silhouette_avg,"\n","_______________________")
    
print(Inertias_list)
print(type(Inertias_list))


# # for explain

# In[3]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from functools import reduce

fig=plt.figure(figsize=(16,10))

x,y = make_blobs(n_samples=1000,n_features=4,centers=[[0,7],[2,2],[4,3],[6,6],[7,7],[10,9],[12,12]],cluster_std=[1.7,0.4,0.9,1.0,1.2,0.1,1.4],random_state=10)

#model= KMeans(n_clusters=4, random_state=10)

model = AffinityPropagation(damping=0.9)

model.fit(x)

y_predict = model.predict(x)


kind,count=np.unique(model.labels_,return_counts=True)
print(kind)
print(count)
print(sum(count))

p1=fig.add_subplot(221)
plt.scatter(x[:,0],x[:,1],c="black")

p2=fig.add_subplot(222)
plt.scatter(x[:,0],x[:,1],c=y_predict)



print(model.predict((x[:30,:])))
print(metrics.calinski_harabasz_score(x,y_predict))
print(model.cluster_centers_)
print(model.inertia_)
print(metrics.silhouette_score(x,y_predict))


# Questions and problems:
# 应用领域的分析？
# 动态抽样，兼顾过程的稳定性， 变异能力过程评价
# 过程能力，信息系统， 嵌入执行。部分产品
# 打通来料到过程  SPC MSA   嵌入到可靠性？
# 
# 主线： 产品改进为主线，质量持续提升。
# 切入点：质量痛点？ 哪些？
# 
# 数据基础薄弱，元数据状态？
# 现有数据资源目标导向，有哪些目标可以完成闭环？
# 
# 
# 蒋部长提出双路径，以及路径相关资源的依赖程度：
# 路径1： 信息化系统关联。
# 路径2： 标准化数据。
# 以及必须要做的工作：
# 基于元数据状态，确定分析维度与准确度。
# 维度对应的准确度（confusion matrix）。元数据状态和方法同步寻找？
# 划定范围。
# 
# 
# 讨论目的：
# 确定目标：突破头脑风暴法对对应维度无数据支撑的状态，无法对下一步质量问题解决对应的资源投入方向和方式。
# 讨论产出：流程（数据成熟度）与算法（对应目标确定方式和参数）。
# 
# 
# 绕不过的点：
# 
# 元数据状态 算法准确率数反推据利用率，（可以反推， 出具成熟度标准。 但是准确率提升程度评估不了。模拟数据解决不了Y。）
# 
# 产品和项点分类（X量级， 执行范围？）
# 
# 资源和产品。目标是针对资源配置还是产品改善？数据源工作人员工作方式改善？
# 
# 
# 
# 

# In[ ]:


from pyclust import KMedoids
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

