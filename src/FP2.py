#!/usr/bin/env python
"""! @brief This file contains the code for augmenting the data contained in a csv file
"""


## 
# @mainpage SmishGuard: A Smishing Detection SMS Framework 
#
# @section project_description A machine learning model that predicts whether a URL is malicious or genuine one 
#

##
# @file FP2.py 
#
# @brief A python program to read the dataset of CSV format and augmenting the data contained in it 
#
# @section libraries_main Libraries/Modules
# - warnings Library(https://docs.python.org/3/library/warnings.html#:~:text=Warning%20messages%20are%20typically%20issued,program%20uses%20an%20obsolete%20module.)
# - pandas Library(https://pandas.pydata.org/docs/) 
# - textaugment Library(https://github.com/dsfsi/textaugment)
# - nlpaug Library(https://towardsdatascience.com/data-augmentation-library-for-text-9661736b13ff)
#
# @section author_code Author
# - Created by Sreeraj R S on 15/02/2023
# - Modified by Sreeraj R S on 28/02/2023
#

# Imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
df=pd.read_csv("data.csv")
from textaugment import Wordnet
from textaugment import EDA
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

# Returns the no. of rows and columns of a dataframe
df.shape


# Obtains the no. of distinct values of the 'LABEL' column
df['LABEL'].value_counts()


# This code creates new DataFrames by selecting only those rows from an existing DataFrame df where the value in the LABEL column is equal to 1 and 2. It then selects the TEXT column from these rows and converts the data type to a string using the astype() method. Finally, it resets the index of the new DataFrames to start at 0.
df2 = pd.DataFrame(df.query('LABEL==1')['TEXT'].astype('string')).reset_index(drop=True)
df3 = pd.DataFrame(df.query('LABEL==2')['TEXT'].astype('string')).reset_index(drop=True)
t1=Wordnet()
t2=EDA()
aug=nac.OcrAug()
aug2=nac.KeyboardAug()
aug3=nac.RandomCharAug(action='delete')
aug4=naw.SynonymAug(aug_src='wordnet')
aug5=naw.AntonymAug()
aug6=naw.RandomWordAug(action='swap')


for i in df2.index:
    new_row=[]
    text=df2.loc[i]['TEXT']
    for _ in range(0,6):
        text1=t2.random_insertion(text)
        new_row={'LABEL':1,'TEXT':text1}
        df=df.append(pd.Series(new_row,index=df.columns),ignore_index=True)
        
        
for i in df3.index:
    new_row1=[]
    text=df3.loc[i]['TEXT']
    for _ in range(0,9):
        text1=t2.random_insertion(text)
        new_row1={'LABEL':2,'TEXT':text1}
        df=df.append(pd.Series(new_row1,index=df.columns),ignore_index=True)


df.to_csv('data_2.csv',index=False)
