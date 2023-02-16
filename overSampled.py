import pandas as pd

docs = pd.read_csv('data.csv')

class_count = docs.LABEL.value_counts()

x = docs.TEXT
y = docs.LABEL

#random oversampling
from imblearn.over_sampling import RandomOverSampler

sampler = RandomOverSampler(sampling_strategy = 'not majority')
x_res, y_res = sampler.fit_resample(x.values.reshape(-1,1), y)

df1 = pd.DataFrame(x_res, columns=['TEXT'])

result = pd.concat([df1, y_res],axis=1, join="inner")

result.to_csv('oversam_data1.csv', index=False, mode='w')
#random oversampling


x1 = result.TEXT
y1 = result.LABEL

from sklearn.model_selection  import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1,test_size = 0.20, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer()
vect.fit(x1_train)
vect = TfidfVectorizer(stop_words='english')
vect.fit(x)

x_train_transformed = vect.transform(x1_train)
x_test_transformed =vect.transform(x1_test)

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

model = SVC()
model.fit(x_train_transformed,y1_train)

y_pred = model.predict(x_test_transformed)
print("SVM = ",accuracy_score(y1_test, y_pred))


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_train_transformed,y1_train)

predictions = model.predict(x_test_transformed)
print("Decision Tree = ",accuracy_score(y1_test, predictions))


import xgboost as xgb

model = xgb.XGBClassifier()
model.fit(x_train_transformed,y1_train)

predictions = model.predict(x_test_transformed)
print("Extreme Gradient Boost = ",accuracy_score(y1_test, predictions))


from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(x_train_transformed,y1_train)

predictions = model.predict(x_test_transformed)
print("Gradient Boosting Decision Tree = ",accuracy_score(y1_test, predictions))


from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier()
model.fit(x_train_transformed,y1_train)

predictions = model.predict(x_test_transformed)
print("Bagging Classifier = ",accuracy_score(y1_test, predictions))


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(x_train_transformed,y1_train)

predictions = model.predict(x_test_transformed)
print("K-Nearest Neighbour = ",accuracy_score(y1_test, predictions))


from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(x_train_transformed,y1_train)

predictions = model.predict(x_test_transformed)
print("Extra Tree = ",accuracy_score(y1_test, predictions))


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x_train_transformed,y1_train)

predictions = model.predict(x_test_transformed)
print("Random Forest = ",accuracy_score(y1_test, predictions))


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train_transformed,y1_train)

predictions = model.predict(x_test_transformed)
print("Logistic Regression = ",accuracy_score(y1_test, predictions))


from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(x_train_transformed,y1_train)

predictions = model.predict(x_test_transformed)
print("Adaptive Boost = ",accuracy_score(y1_test, predictions))


from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB(binarize = True)
model.fit(x_train_transformed,y1_train)

predictions = model.predict(x_test_transformed)
print("Bernouli NB = ",accuracy_score(y1_test, predictions))


from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(x_train_transformed.toarray(),y1_train)

predictions = model.predict(x_test_transformed.toarray())
print("Gaussian NB = ",accuracy_score(y1_test, predictions))


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train_transformed,y1_train)

predictions = model.predict(x_test_transformed)
print("Multinomial NB = ",accuracy_score(y1_test, predictions))











