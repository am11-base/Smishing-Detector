#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas_profiling import ProfileReport 
import seaborn as sns


# In[23]:


df=pd.read_csv('data.csv')
profile=ProfileReport(df)
profile.to_file(output_file="data.html")


# In[24]:


df['TEXT'].nunique()


# In[25]:


df.info()


# In[26]:


pd.get_dummies(df,['TEXT']).head()


# In[27]:


pd.get_dummies(df,['TEXT']).info()


# In[28]:


corelation=df.corr()


# In[29]:


sns.heatmap(corelation,xticklabels=corelation.columns,yticklabels=corelation.columns,annot=True)
plt.show()

