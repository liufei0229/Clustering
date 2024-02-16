#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
 
n_samples = [10, 10, 10, 10]
centers = [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [6.0, 6.0]]
cluster_std = [0.4, 0.4, 0.4, 0.4]

x, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=0, shuffle=False)


print(x)


# In[2]:


print(y)


# In[3]:


clf = svm.SVC(kernel='linear', C=10, gamma=0.5, decision_function_shape='ovo')
clf.fit(x, y)


# In[ ]:




