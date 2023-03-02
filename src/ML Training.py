#!/usr/bin/env python
# coding: utf-8

# In[100]:


import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')
import re
import string
df0=pd.read_csv("data.csv")
df = pd.read_csv("data_2.csv")
df0.info()


# In[101]:


stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.WordNetLemmatizer()


# In[102]:


def clean_text(text):
    no_punctuation = ''.join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    stems = [ps.lemmatize(word) for word in tokens if word not in stopwords] # Remove Stopwords
    return stems


# In[103]:


print(sorted(Counter(df['LABEL']).items()))


# In[104]:


X=df["TEXT"].astype(str).str.lower()
Y=df["LABEL"]


# In[105]:


#X = [str(x) for x in X] #converting to iterable object before passing to fit transform
#vect = TfidfVectorizer(analyzer=clean_text)
#vector=vect.fit_transform(X)
#X=pd.DataFrame(vector.toarray())


# In[106]:


split_index = 6042
X_train, X_test, Y_train, Y_test = train_test_split(X[:split_index], Y[:split_index],random_state=42)


# In[107]:


X_train = X[split_index:]
Y_train = Y[split_index:]


# In[108]:


print(X.head())


# In[109]:


vectorizer=TfidfVectorizer()
X_train_tfidf=vectorizer.fit_transform(X_train)
clf=LogisticRegression()
clf.fit(X_train_tfidf, Y_train)


# In[110]:


X_test_tfidf = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))


# In[ ]:




