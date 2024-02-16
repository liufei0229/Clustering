#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #数维度数组与矩阵运算
import xlrd #读取excel
import pandas as pd #分析结构化数据
from matplotlib import pyplot as plt #可视化

io1 = r'E:/纽约地铁/已处理问题数据.xlsx'
df1 = pd.read_excel(io1, sheet_name = "learn")
df1.head()


# In[2]:


#标准化
x_data=df1.loc[:,["gd" ,"ysg"]]

from sklearn import preprocessing
#Z-Score标准化
#建立StandardScaler对象
zscore = preprocessing.StandardScaler()

# 标准化处理
df1_zs_array = zscore.fit_transform(x_data)
df1_zs=pd.DataFrame(df1_zs_array,columns=["gd_zs" ,"ysg_zs"])
df1_zs.head()


# In[3]:


data=pd.concat([df1_zs,df1["14_07"],df1["14_08"],df1["14_09"],df1["14_10"],df1["14_11"],df1["15_05"],df1["15_06"],df1["15_07"],df1["15_08"],df1["15_09"],df1["16_08"],df1["16_09"],df1["16_10"],df1["unqualified"]],axis=1)

#验证训练集
print(data.shape)
data.head()


# In[4]:


#划分数据与标签
import sklearn
from sklearn.model_selection import train_test_split
x,y=np.split(data,indices_or_sections=(15,),axis=1) #x为数据，y为标签
train_data,test_data,train_label,test_label =train_test_split(x,y, random_state=7, train_size=0.7,test_size=0.3)
print(train_data.shape)


# In[13]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(class_weight='balanced',max_depth=3, n_estimators=100, random_state=7)
classifier.fit(train_data,train_label.values.ravel())


# In[14]:


import sklearn.model_selection as ms
import sklearn.metrics as sm
# 精确度
score = ms.cross_val_score(classifier, train_data, train_label.values.ravel(),cv=4, scoring='accuracy')
print('accuracy score=', score)
print('accuracy mean=', score.mean())

# 查准率
score = ms.cross_val_score(classifier, train_data, train_label.values.ravel(), cv=4, scoring='precision_weighted')
print('precision_weighted score=', score)
print('precision_weighted mean=', score.mean())

# 召回率
score = ms.cross_val_score(classifier, train_data, train_label.values.ravel(), cv=4, scoring='recall_weighted')
print('recall_weighted score=', score)
print('recall_weighted mean=', score.mean())

# f1得分
score = ms.cross_val_score(classifier, train_data, train_label.values.ravel(), cv=4, scoring='f1_weighted')
print('f1_weighted score=', score)
print('f1_weighted mean=', score.mean())  


# In[15]:


# calculate the accuracy
from sklearn.metrics import accuracy_score
train_label_pre=classifier.predict(train_data) #训练集的预测标签
test_label_pre=classifier.predict(test_data) #测试集的预测标签
print("训练集：", accuracy_score(train_label,train_label_pre) )
print("测试集：", accuracy_score(test_label,test_label_pre) )


# In[16]:


#以下均按照测试集进行计算
# 获取混淆矩阵
m = sm.confusion_matrix(test_label,test_label_pre)
print('混淆矩阵如下：', m, sep='\n')

# 获取分类报告
r = sm.classification_report(test_label, test_label_pre)
print('分类报告如下：', r, sep='\n')


# In[17]:


import seaborn as sn
sn.heatmap(sm.confusion_matrix(train_label, train_label_pre), annot=True)


# In[18]:


sn.heatmap(sm.confusion_matrix(test_label, test_label_pre), annot=True)


# In[31]:


from sklearn.metrics import roc_curve, auc  ###计算roc和auc
#通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
y_score = classifier.fit(train_data,train_label.values.ravel()).predict_proba(test_data)
 
# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(test_label, y_score[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
 
plt.figure()
lw = 2
plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (AUC area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Reciever Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()


# In[32]:


#获取特征重要性比例
classifier.feature_importances_


# In[33]:


name_list_pre=df1.columns.values
print(name_list_pre)


# In[34]:


df2=df1.drop(labels=["gap","unqualified"],axis=1)
name_list=df2.columns.values
print(name_list)


# In[35]:


import matplotlib.pyplot as plt
 
num_list = classifier.feature_importances_
plt.figure(figsize=(10,10))
plt.xticks(num_list,name_list,size='small',rotation=30)
plt.bar(range(len(num_list)), num_list,tick_label=name_list)
plt.show()


# In[36]:


feature_importance=np.row_stack([name_list,classifier.feature_importances_])
pd.DataFrame(feature_importance)


# In[82]:


df1.describe()


# In[57]:


xdata_1=df1.loc[:,["gd" ,"ysg","14_10","15_06","15_07","15_08"]]
xdata_1.head()


# In[58]:


ydata_1=df1.loc[:,["unqualified"]]
ydata_1.head()


# In[59]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(xdata_1,ydata_1,test_size=0.3,random_state=16)


# In[63]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression(class_weight='balanced')
model.fit(X=x_train,y=y_train.values.ravel())


# In[71]:


y_test_pred=model.predict(X=x_test)# 默认阀值为0.5
y_test_pred_proba=model.predict_proba(X=x_test) # 可以自定义阀值，比如自定义阀值0.6
print(y_test_pred)
print(y_test_pred_proba)


# In[72]:


#该方法可以自行调整阈值，并获得新阈值下的分类
def thes_func(x):
    thes=0.3
    return 1 if x>thes else 0
y_test_pred_thes=list(map(thes_func,y_test_pred_proba[:,1]))
print(y_test_pred_thes)


# In[73]:


model.coef_


# In[74]:


model.intercept_


# In[70]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_true=y_test,y_pred=y_test_pred)
print('Accuracy:{}'.format(acc))


# In[76]:


print(model.predict_proba(xdata_1))


# In[ ]:




