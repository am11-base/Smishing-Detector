#!/usr/bin/env python
# coding: utf-8

# In[244]:
## 
# @mainpage SmishGuard: A Smishing Detection SMS Framework 
#
# @section author_code Author
# - Created by Sreeraj R S on 15/02/2023
# - Modified by Sreeraj R S on 28/02/2023
#
# @section data_augmentation Data Augmentation
#
# @section vector_ization Vectorization 
# 
# 

##
# @file FP2_ham.py 
#
# @brief A python program to read the dataset of CSV format and augmenting the data contained in it 
#
## @section code_block Blocks
#
## @subsection libraries_main Libraries
# - warnings Library(https://docs.python.org/3/library/warnings.html#:~:text=Warning%20messages%20are%20typically%20issued,program%20uses%20an%20obsolete%20module.)
# - pandas Library(https://pandas.pydata.org/docs/) 
# - textaugment Library(https://github.com/dsfsi/textaugment)
# - nlpaug Library(https://towardsdatascience.com/data-augmentation-library-for-text-9661736b13ff)
#
## @code{.py}
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
from textaugment import Wordnet
from textaugment import EDA
## @endcode
#
df=pd.read_csv("data.csv")


# In[246]:

## Documentation for shape
#
# More details
df.shape


# In[247]:


df['LABEL'].value_counts()


# In[248]:


df1 = pd.DataFrame(df.query('LABEL==0')['TEXT'].astype('string')).reset_index(drop=True)
df2 = pd.DataFrame(df.query('LABEL==1')['TEXT'].astype('string')).reset_index(drop=True)
df3 = pd.DataFrame(df.query('LABEL==2')['TEXT'].astype('string')).reset_index(drop=True)


# In[249]:



t1=Wordnet()
t2=EDA()


# In[250]:


import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
aug=nac.OcrAug()
aug2=nac.KeyboardAug()
aug3=nac.RandomCharAug(action='delete')
aug4=naw.SynonymAug(aug_src='wordnet')
aug5=naw.AntonymAug()
aug6=naw.RandomWordAug(action='swap')


# In[251]:


for i in df1.index:
    new_row=[]
    text=df1.loc[i]['TEXT']
    for _ in range(0,1):
        #text1=t1.augment(text)
        #text1=t2.random_insertion(text)
        #text1=t2.random_deletion(text)
        #text1=aug.augment(text)
        #text1=aug2.augment(text)
        #text1=aug3.augment(text)
        #text1=aug4.augment(text)
        text1=aug5.augment(text)
        #new_row={'LABEL': 0, 'TEXT': text1} 
        new_row={'LABEL': 0, 'TEXT': text1[0]} #variation for appending nlpaug as it adds square brackets
        df=df.append(pd.Series(new_row,index=df.columns),ignore_index=True)


# In[252]:


for i in df2.index:
    new_row=[]
    text=df2.loc[i]['TEXT']
    for _ in range(0,13):
        #text1=t1.augment(text)
        #text1=t2.random_insertion(text)
        #text1=t2.random_deletion(text)
        #text1=aug.augment(text)
        #text1=aug2.augment(text)
        #text1=aug3.augment(text)
        #text1=aug4.augment(text)
        text1=aug5.augment(text)
        new_row1={'LABEL': 1, 'TEXT': text1[0]}
        df=df.append(pd.Series(new_row1,index=df.columns),ignore_index=True)


# In[253]:


for i in df3.index:
    new_row1=[]
    text=df3.loc[i]['TEXT']
    for _ in range(0,18):
        #text1=t1.augment(text)
        #text1=t2.random_insertion(text)
        #text1=t2.random_deletion(text)
        #text1=aug.augment(text)
        #text1=aug2.augment(text)
        #text1=aug3.augment(text)
        #text1=aug4.augment(text)
        text1=aug5.augment(text)
        new_row2={'LABEL': 2, 'TEXT': text1[0]}
        df=df.append(pd.Series(new_row2,index=df.columns),ignore_index=True)


# In[254]:


df.to_csv('data_8.csv',index=False)

