#!/usr/bin/env python
# coding: utf-8

# In[16]:


import gensim
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
import pandas as pd
import numpy as np
import gensim.downloader as api


# In[17]:


wv=api.load('glove-twitter-50')


# In[18]:


wv.save("vectors.kv")


# In[19]:


wv['apple']


# In[20]:


from gensim.models import KeyedVectors
wv=KeyedVectors.load("vectors.kv")


# In[23]:


df=pd.read_csv("data_2.csv")


# In[24]:


def sentvec(sent):
    v_size=wv.vector_size
    wv_res=np.zeros(v_size)
    ctr=1
    for w in sent:
        if w in wv:
            ctr+=1
            wv_res+=wv[w]
    wv_res/=ctr
    return wv_res


# In[27]:


# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    doc = nlp(sentence)



    # print(doc)
    # print(type(doc))

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() for word in doc ]

    # print(mytokens)

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens


# In[29]:


import spacy
nlp=spacy.load("en_core_web_sm")
stop_words=nlp.Defaults.stop_words
#print(stop_words)


# In[30]:


import string
punctuations = string.punctuation
print(punctuations)


# In[37]:


df['TOKENS'] = df['TEXT'].apply(spacy_tokenizer)


# In[39]:


df['VEC'] = df['TOKENS'].apply(sentvec)


# In[44]:


X = df['VEC'].to_list()
Y = df['LABEL'].to_list()


# In[45]:


split_index = 6042
X_train=X[split_index:]
Y_train=Y[split_index:]
X_test = X[:split_index]
Y_test = Y[:split_index]


# In[46]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
# fit
classifier.fit(X_train,Y_train)


# In[48]:


from sklearn import metrics
predicted = classifier.predict(X_test)
print("Logistic Regression Accuracy:",metrics.accuracy_score(Y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(Y_test, predicted,average='micro'))
print("Logistic Regression Recall:",metrics.recall_score(Y_test, predicted,average='micro'))

