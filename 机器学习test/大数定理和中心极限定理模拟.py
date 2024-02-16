#!/usr/bin/env python
# coding: utf-8

# 大数定理

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
all_value = np.random.randint(1,10000,5000)
sample_size = []
sample_maen = []
for i in range(10,10000,10):
    sample_size.append(i)
    sample_maen.append(np.random.choice(all_value,i).mean())

pd.DataFrame({"sample_size":sample_size,"sample_maen":sample_maen}).set_index("sample_size").plot(color = "grey")
plt.axhline(all_value.mean(),color = "red")


# 中心极限定理

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
random_data = np.random.randint(1, 1000, 10000)
samples_mean = []
for i in range(100000):
    sample=np.random.choice(random_data,5000)
    samples_mean.append(sample.mean())
samples_mean=np.array( samples_mean)
plt.hist( samples_mean,bins=50,color='blue')
plt.grid()
plt.show()


# In[ ]:





# In[ ]:




