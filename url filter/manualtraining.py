
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
#data = pd.read_csv(r'E:\Users\hashim\Documents\cse 19-23\4th yr\project\references\features.csv')
data.drop(['url'],axis = 1,inplace = True)
X=data.drop('class_label', axis=1)
y= data['class_label']
# print(y.head())
# print(X.head())

from sklearn.model_selection  import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y,test_size = 0.20, random_state=26)


model=RandomForestClassifier()

# fit
naive_model = model.fit(x_train,y_train)

# predict class
y_pred_class = model.predict(x_test)

# printing the overall accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))