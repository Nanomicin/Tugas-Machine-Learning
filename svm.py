import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import pandas as pd

df = df = pd.read_csv('dataset.csv', nrows=300)

X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

print(">> Finish Reading dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print(">> Finish Split Data 80% Train 20% Test")

clf = svm.SVC(kernel='linear', gamma="auto")

print(">> Start Training with Gamma=auto")
clf.fit(X_train, y_train)
print(">> Finish Training")

y_pred = clf.predict(X_test)
print(">> Finish Prediction")

print(">> Result (Precision, Recall, F-Measure) : ")
result = classification_report(y_test, y_pred)
print(result)
