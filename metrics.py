#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

docs = pd.read_csv('data_8.csv')

docs.head()


# In[2]:


class_count=docs.LABEL.value_counts()
class_count


# In[3]:


x = docs.TEXT
y = docs.LABEL
print(x.shape)
print(y.shape)


# In[4]:


from sklearn.model_selection  import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.20, random_state=26)
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
vect.fit(x_train)
vect = TfidfVectorizer(stop_words='english')
vectorizer = vect.fit(x)
x_train_transformed = vect.transform(x_train)
x_test_transformed =vect.transform(x_test)
#print(type(x_train_transformed))
#print(x_train_transformed)


# In[12]:


from sklearn.svm import SVC

model = SVC(kernel = 'linear', random_state = 10,verbose= True)
model.fit(x_train_transformed,y_train)


y_pred_class = model.predict(x_test_transformed)


# In[43]:


print("\n")
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class), metrics.f1_score(y_test, y_pred_class, average='macro'), metrics.precision_score(y_test, y_pred_class, average='macro'), metrics.recall_score(y_test, y_pred_class, average='macro'))


# In[7]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_train_transformed,y_train)

y_pred_class = model.predict(x_test_transformed)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class), metrics.f1_score(y_test, y_pred_class, average='macro'), metrics.precision_score(y_test, y_pred_class, average='macro'), metrics.recall_score(y_test, y_pred_class, average='macro'))


# In[10]:


import xgboost as xgb

model = xgb.XGBClassifier()
model.fit(x_train_transformed,y_train)

y_pred_class = model.predict(x_test_transformed)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class), metrics.f1_score(y_test, y_pred_class, average='macro'), metrics.precision_score(y_test, y_pred_class, average='macro'), metrics.recall_score(y_test, y_pred_class, average='macro'))


# In[14]:


from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(x_train_transformed,y_train)

y_pred_class = model.predict(x_test_transformed)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class), metrics.f1_score(y_test, y_pred_class, average='macro'), metrics.precision_score(y_test, y_pred_class, average='macro'), metrics.recall_score(y_test, y_pred_class, average='macro'))


# In[16]:


from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier()
model.fit(x_train_transformed,y_train)

y_pred_class = model.predict(x_test_transformed)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class), metrics.f1_score(y_test, y_pred_class, average='macro'), metrics.precision_score(y_test, y_pred_class, average='macro'), metrics.recall_score(y_test, y_pred_class, average='macro'))


# In[23]:


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(x_train_transformed,y_train)

y_pred_class = model.predict(x_test_transformed)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class), metrics.f1_score(y_test, y_pred_class, average='macro'), metrics.precision_score(y_test, y_pred_class, average='macro'), metrics.recall_score(y_test, y_pred_class, average='macro'))


# In[25]:


from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(x_train_transformed,y_train)

y_pred_class = model.predict(x_test_transformed)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class), metrics.f1_score(y_test, y_pred_class, average='macro'), metrics.precision_score(y_test, y_pred_class, average='macro'), metrics.recall_score(y_test, y_pred_class, average='macro'))


# In[27]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x_train_transformed,y_train)

y_pred_class = model.predict(x_test_transformed)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class), metrics.f1_score(y_test, y_pred_class, average='macro'), metrics.precision_score(y_test, y_pred_class, average='macro'), metrics.recall_score(y_test, y_pred_class, average='macro'))


# In[29]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train_transformed,y_train)

y_pred_class = model.predict(x_test_transformed)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class), metrics.f1_score(y_test, y_pred_class, average='macro'), metrics.precision_score(y_test, y_pred_class, average='macro'), metrics.recall_score(y_test, y_pred_class, average='macro'))


# In[31]:


from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(x_train_transformed,y_train)

y_pred_class = model.predict(x_test_transformed)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class), metrics.f1_score(y_test, y_pred_class, average='macro'), metrics.precision_score(y_test, y_pred_class, average='macro'), metrics.recall_score(y_test, y_pred_class, average='macro'))


# In[35]:


from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()
model.fit(x_train_transformed,y_train)

y_pred_class = model.predict(x_test_transformed)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class), metrics.f1_score(y_test, y_pred_class, average='macro'), metrics.precision_score(y_test, y_pred_class, average='macro'), metrics.recall_score(y_test, y_pred_class, average='macro'))


# In[39]:


from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(x_train_transformed.toarray(),y_train)

y_pred_class = model.predict(x_test_transformed.toarray())


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class), metrics.f1_score(y_test, y_pred_class, average='macro'), metrics.precision_score(y_test, y_pred_class, average='macro'), metrics.recall_score(y_test, y_pred_class, average='macro'))


# In[42]:


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train_transformed,y_train)

y_pred_class = model.predict(x_test_transformed)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class), metrics.f1_score(y_test, y_pred_class, average='macro'), metrics.precision_score(y_test, y_pred_class, average='macro'), metrics.recall_score(y_test, y_pred_class, average='macro'))

