#!/usr/bin/env python
# coding: utf-8

# # TF-IDF 

# In[3]:


import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
import gensim
import gensim.downloader as api
import nltk
import re
import string


# In[4]:


nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')
#df0=pd.read_csv("data.csv")
df = pd.read_csv("data_2.csv")
#df0.info()


# In[5]:


stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.WordNetLemmatizer()


# In[6]:


def clean_text(text):
    no_punctuation = ''.join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    stems = [ps.lemmatize(word) for word in tokens if word not in stopwords] # Remove Stopwords
    return stems


# In[7]:


print(sorted(Counter(df['LABEL']).items()))


# In[8]:


X=df['TEXT']
Y=df['LABEL']


# In[9]:


print(X.shape)
print(Y.shape)


# In[10]:


split_index = 6042


# In[11]:


X_train=X[split_index:]
Y_train=Y[split_index:]


# In[12]:


X_train.shape


# In[13]:


X_test = X[:split_index]
Y_test = Y[:split_index]


# In[14]:


X_test.shape


# In[15]:


vect = TfidfVectorizer(analyzer=clean_text)
vector=vect.fit_transform(X_train)
X_train=pd.DataFrame(vector.toarray())


# In[16]:


vector1=vect.transform(X_test)
X_test=pd.DataFrame(vector1.toarray())


# In[17]:


print(X_train.shape)
print(X_test.shape)


# In[18]:


#X = [str(x) for x in X] #converting to iterable object before passing to fit transform
#vect = TfidfVectorizer(analyzer=clean_text)
#vector=vect.fit_transform(X)
#X=pd.DataFrame(vector.toarray())


# In[19]:


from sklearn import metrics


# In[34]:


classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
print(X.shape)
print(Y.shape)
classifier.fit(X_train,Y_train)
predicted = classifier.predict(X_test)
pred_prob=classifier.predict_proba(X_test)
print(metrics.accuracy_score(Y_test, predicted))
print(metrics.roc_auc_score(Y_test,pred_prob,multi_class='ovr'))
print(metrics.f1_score(Y_test,predicted,average='micro'))


# In[35]:


classifier = MultinomialNB()

classifier.fit(X_train,Y_train)
predicted = classifier.predict(X_test)
pred_prob=classifier.predict_proba(X_test)
print(metrics.accuracy_score(Y_test, predicted))
print(metrics.roc_auc_score(Y_test,pred_prob,multi_class='ovr'))
print(metrics.f1_score(Y_test,predicted,average='micro'))


# In[36]:


classifier=RandomForestClassifier()
classifier.fit(X_train,Y_train)
predicted = classifier.predict(X_test)
pred_prob=classifier.predict_proba(X_test)
print(metrics.accuracy_score(Y_test, predicted))
print(metrics.roc_auc_score(Y_test,pred_prob,multi_class='ovr'))
print(metrics.f1_score(Y_test,predicted,average='micro'))


# In[37]:


from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train,Y_train)
predicted = classifier.predict(X_test)
pred_prob=classifier.predict_proba(X_test)
print(metrics.accuracy_score(Y_test, predicted))
print(metrics.roc_auc_score(Y_test,pred_prob,multi_class='ovr'))
print(metrics.f1_score(Y_test,predicted,average='micro'))


# In[ ]:


from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB(binarize = True)
classifier.fit(X_train,Y_train)
predicted = classifier.predict(X_test)
pred_prob=classifier.predict_proba(X_test)
print(metrics.accuracy_score(Y_test, predicted))
print(metrics.roc_auc_score(Y_test,pred_prob,multi_class='ovr'))
print(metrics.f1_score(Y_test,predicted,average='micro'))


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

classifier = AdaBoostClassifier()
classifier.fit(X_train,Y_train)
predicted = classifier.predict(X_test)
pred_prob=classifier.predict_proba(X_test)
print(metrics.accuracy_score(Y_test, predicted))
print(metrics.roc_auc_score(Y_test,pred_prob,multi_class='ovr'))
print(metrics.f1_score(Y_test,predicted,average='micro'))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()
classifier.fit(X_train,Y_train)
predicted = classifier.predict(X_test)
pred_prob=classifier.predict_proba(X_test)
print(metrics.accuracy_score(Y_test, predicted))
print(metrics.roc_auc_score(Y_test,pred_prob,multi_class='ovr'))
print(metrics.f1_score(Y_test,predicted,average='micro'))


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier

classifier= ExtraTreesClassifier()
classifier.fit(X_train,Y_train)
predicted = classifier.predict(X_test)
pred_prob=classifier.predict_proba(X_test)
print(metrics.accuracy_score(Y_test, predicted))
print(metrics.roc_auc_score(Y_test,pred_prob,multi_class='ovr'))
print(metrics.f1_score(Y_test,predicted,average='micro'))


# In[ ]:


from sklearn.ensemble import BaggingClassifier

classifier = BaggingClassifier()
classifier.fit(X_train,Y_train)
predicted = classifier.predict(X_test)
pred_prob=classifier.predict_proba(X_test)
print(metrics.accuracy_score(Y_test, predicted))
print(metrics.roc_auc_score(Y_test,pred_prob,multi_class='ovr'))
print(metrics.f1_score(Y_test,predicted,average='micro'))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

classifier= GradientBoostingClassifier()
classifier.fit(X_train,Y_train)
predicted = classifier.predict(X_test)
pred_prob=classifier.predict_proba(X_test)
print(metrics.accuracy_score(Y_test, predicted))
print(metrics.roc_auc_score(Y_test,pred_prob,multi_class='ovr'))
print(metrics.f1_score(Y_test,predicted,average='micro'))


# In[ ]:


import xgboost as xgb

classifier= xgb.XGBClassifier()
classifier.fit(X_train,Y_train)
predicted = classifier.predict(X_test)
pred_prob=classifier.predict_proba(X_test)
print(metrics.accuracy_score(Y_test, predicted))
print(metrics.roc_auc_score(Y_test,pred_prob,multi_class='ovr'))
print(metrics.f1_score(Y_test,predicted,average='micro'))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

classifier= DecisionTreeClassifier()
classifier.fit(X_train,Y_train)
predicted = classifier.predict(X_test)
pred_prob=classifier.predict_proba(X_test)
print(metrics.accuracy_score(Y_test, predicted))
print(metrics.roc_auc_score(Y_test,pred_prob,multi_class='ovr'))
print(metrics.f1_score(Y_test,predicted,average='micro'))


# In[ ]:


from sklearn.svm import SVC

classifier= SVC(gamma='auto', probability=True)
classifier.fit(X_train,Y_train)
predicted = classifier.predict(X_test)
pred_prob=classifier.predict_proba(X_test)
print(metrics.accuracy_score(Y_test, predicted))
print(metrics.roc_auc_score(Y_test,pred_prob,multi_class='ovr'))
print(metrics.f1_score(Y_test,predicted,average='micro'))

