#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #数维度数组与矩阵运算
import xlrd #读取excel
import pandas as pd #分析结构化数据
from matplotlib import pyplot as plt #可视化

io1 = r'E:/Python/机器学习test/game_ori.xlsx'
df1 = pd.read_excel(io1, sheet_name = "Sheet1")
df1.head()


sklearn包导入情况

逻辑回归：
#from sklearn.linear_model import LogisticRegression


朴素贝叶斯：
#from sklearn.naive_bayes import GaussianNB

K-近邻：
from sklearn.neighbors import KNeighborsClassifier

决策树：
from sklearn.tree import DecisionTreeClassifier


# 略过数据预处理， 在excel里已经完成，实际可能用M语言。

# In[2]:


#标准化
x_data=df1.loc[:,["last_login_to_now" ,"duration","level_end","os_corr","active_days","avg_session_cnt"]]

from sklearn import preprocessing
#Z-Score标准化
#建立StandardScaler对象
zscore = preprocessing.StandardScaler()

# 标准化处理
df1_zs_array = zscore.fit_transform(x_data)
df1_zs=pd.DataFrame(df1_zs_array,columns=["last_login_to_now_zs" ,"duration_zs","level_end_zs","os_corr_zs","active_days_zs","avg_session_cnt_zs"])
df1_zs.head()


# In[3]:


data=pd.concat([df1_zs,df1["is_payer"]],axis=1)

#验证训练集
print(data.shape)
data.head()

1. split(数据，分割位置，轴=1（水平分割） or 0（垂直分割）)。
2.  sklearn.model_selection.train_test_split随机划分训练集与测试集。train_test_split(train_data,train_label,test_size=数字, random_state=0)
参数解释：

　    　train_data：所要划分的样本特征集

　    　train_label：所要划分的样本类别

　    　test_size：样本占比，如果是整数的话就是样本的数量.(注意：)

                   --  test_size:测试样本占比。 默认情况下，该值设置为0.25。 默认值将在版本0.21中更改。 只有train_size没有指定时， 

                        它将保持0.25，否则它将补充指定的train_size，例如train_size=0.6,则test_size默认为0.4。

                   -- train_size:训练样本占比。

random_state：是随机数的种子。
随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
# In[4]:


#划分数据与标签
import sklearn
from sklearn.model_selection import train_test_split
x,y=np.split(data,indices_or_sections=(6,),axis=1) #x为数据，y为标签
train_data,test_data,train_label,test_label =train_test_split(x,y, random_state=7, train_size=0.7,test_size=0.3)
print(train_data.shape)

kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。

kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。

decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，

decision_function_shape='ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。#train the SVM classifer
from sklearn import svm
classifier=svm.SVC(C=2,kernel='rbf',gamma=8,decision_function_shape='ovo') 
classifier.fit(train_data,train_label.values.ravel()) 
#this csdn wonderfully clarifies random forest
https://blog.csdn.net/qq_29750461/article/details/81516008?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~baidu_landing_v2~default-6-81516008.nonecase&utm_term=python%20%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97sklearn&spm=1000.2123.3001.4430
# In[5]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(max_depth=4, n_estimators=20, random_state=7)
classifier.fit(train_data,train_label.values.ravel())


# In[6]:


import sklearn.model_selection as ms
import sklearn.metrics as sm
# 精确度
score = ms.cross_val_score(classifier, train_data, train_label.values.ravel(),cv=3, scoring='accuracy')
print('accuracy score=', score)
print('accuracy mean=', score.mean())

# 查准率
score = ms.cross_val_score(classifier, train_data, train_label.values.ravel(), cv=3, scoring='precision_weighted')
print('precision_weighted score=', score)
print('precision_weighted mean=', score.mean())

# 召回率
score = ms.cross_val_score(classifier, train_data, train_label.values.ravel(), cv=3, scoring='recall_weighted')
print('recall_weighted score=', score)
print('recall_weighted mean=', score.mean())

# f1得分
score = ms.cross_val_score(classifier, train_data, train_label.values.ravel(), cv=3, scoring='f1_weighted')
print('f1_weighted score=', score)
print('f1_weighted mean=', score.mean())  


# In[7]:


# calculate the accuracy
from sklearn.metrics import accuracy_score
train_label_pre=classifier.predict(train_data) #训练集的预测标签
test_label_pre=classifier.predict(test_data) #测试集的预测标签
print("训练集：", accuracy_score(train_label,train_label_pre) )
print("测试集：", accuracy_score(test_label,test_label_pre) )


# In[8]:


#以下均按照测试集进行计算
# 获取混淆矩阵
m = sm.confusion_matrix(test_label,test_label_pre)
print('混淆矩阵如下：', m, sep='\n')

# 获取分类报告
r = sm.classification_report(test_label, test_label_pre)
print('分类报告如下：', r, sep='\n')

#查看决策函数,SVM有，RF没有
print('train_decision_function:\n',classifier.decision_function(test_data)) 
print('predict_result:\n',classifier.predict(test_data))apply(X)	  Apply trees in the forest to X, return leaf indices.
decision_path(X)	  Return the decision path in the forest
fit(X, y[, sample_weight])	    Build a forest of trees from the training set (X, y).
get_params([deep])	   Get parameters for this estimator.
predict(X)	    Predict class for X.
predict_log_proba(X)	    Predict class log-probabilities for X.
predict_proba(X)	     Predict class probabilities for X.
score(X, y[, sample_weight])	     Returns the mean accuracy on the given test data and labels.
set_params(**params)	    Set the parameters of this estimator.
# In[9]:


lf_indi=pd.DataFrame(classifier.apply(train_data))
lf_indi.describe()


# In[10]:


path=pd.DataFrame(classifier.decision_path(train_data))
a=path.transpose()
print(a[1])


# In[11]:


#获取特征重要性比例
classifier.feature_importances_


# # 以下是实际需要预测的数据

# In[11]:


io2 = r'E:/Python/机器学习test/game_pre.xlsx'
df2 = pd.read_excel(io2, sheet_name = "sheet1")
rows=df2.shape[0]
df=df1.append(df2,sort=False)
print(rows)
df2


# In[12]:


#标准化
x_data_pre=df.loc[:,["last_login_to_now" ,"duration","level_end","os_corr","active_days","avg_session_cnt"]]

from sklearn import preprocessing
#Z-Score标准化
#建立StandardScaler对象
zscore = preprocessing.StandardScaler()
# 标准化处理
df_zs_array = zscore.fit_transform(x_data_pre)
df_zs=pd.DataFrame(df_zs_array,columns=["last_login_to_now_zs" ,"duration_zs","level_end_zs","os_corr_zs","active_days_zs","avg_session_cnt_zs"])
df_zs.head()


# In[13]:


classifier.predict(df_zs)


# In[14]:


pre=pd.DataFrame(classifier.predict(df_zs),columns=["predict_ispayer"])
predict=pre.iloc[-rows:]
predict_df=predict.reset_index(drop=True)
predict_df

#SVM计算概率
possi=pd.DataFrame(classifier.decision_function(df_zs),columns=["possibility"])
possibility=possi.iloc[-rows:]
possibility_df=possibility.reset_index(drop=True) 
possibility_df
# In[15]:


possi=pd.DataFrame(classifier.predict_proba(df_zs),columns=["possibility=0","possibility=1"])
possibility=possi.iloc[-rows:]
possibility_df=possibility.reset_index(drop=True) 
possibility_df


# In[16]:


df_pre=pd.concat([df2,possibility_df,predict_df],axis=1)
df_pre


# In[ ]:





# In[ ]:




