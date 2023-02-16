import pandas as pd

docs = pd.read_csv('data.csv')

class_count = docs.LABEL.value_counts()

x = docs.TEXT
y = docs.LABEL

from sklearn.model_selection  import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.20, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
vect.fit(x_train)
vect = TfidfVectorizer(stop_words='english')
vect.fit(x)

x_train_transformed = vect.transform(x_train)
x_test_transformed =vect.transform(x_test)

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

model = SVC()
model.fit(x_train_transformed,y_train)

y_pred = model.predict(x_test_transformed)
print("SVM = ",accuracy_score(y_test, y_pred))


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_train_transformed,y_train)

predictions = model.predict(x_test_transformed)
print("Decision Tree = ",accuracy_score(y_test, predictions))


import xgboost as xgb

model = xgb.XGBClassifier()
model.fit(x_train_transformed,y_train)

predictions = model.predict(x_test_transformed)
print("Extreme Gradient Boosting = ",accuracy_score(y_test, predictions))


from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(x_train_transformed,y_train)

predictions = model.predict(x_test_transformed)
print("Gradient Boosting Decision Tree = ",accuracy_score(y_test, predictions))


from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier()
model.fit(x_train_transformed,y_train)

predictions = model.predict(x_test_transformed)
print("Bagging Classifier = ",accuracy_score(y_test, predictions))


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(x_train_transformed,y_train)

predictions = model.predict(x_test_transformed)
print("K-Nearest Neighbour = ",accuracy_score(y_test, predictions))


from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(x_train_transformed,y_train)

predictions = model.predict(x_test_transformed)
print("Extra Tree = ",accuracy_score(y_test, predictions))


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x_train_transformed,y_train)

predictions = model.predict(x_test_transformed)
print("Random Forest = ",accuracy_score(y_test, predictions))


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train_transformed,y_train)

predictions = model.predict(x_test_transformed)
print("Logistic Regression = ",accuracy_score(y_test, predictions))


from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(x_train_transformed,y_train)

predictions = model.predict(x_test_transformed)
print("Adaptive Boost = ",accuracy_score(y_test, predictions))


from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB(binarize = True)
model.fit(x_train_transformed,y_train)

predictions = model.predict(x_test_transformed)
print("Bernouli NB = ",accuracy_score(y_test, predictions))


from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(x_train_transformed.toarray(),y_train)

predictions = model.predict(x_test_transformed.toarray())
print("Gaussian NB = ",accuracy_score(y_test, predictions))


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train_transformed,y_train)

predictions = model.predict(x_test_transformed)
print("Multinomial NB = ",accuracy_score(y_test, predictions))


