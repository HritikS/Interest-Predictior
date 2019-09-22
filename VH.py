import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle

df = pd.read_csv('data.csv')
y = df['P'].values
X = df[['A', 'B','C', 'D', 'E', 'F', 'G', 'H','I', 'J', 'K', 'L', 'M', 'N', 'O']] .values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
neigh = LogisticRegression(solver='lbfgs', multi_class='multinomial').fit(X_train,y_train)
yhat = neigh.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
saved_model = pickle.dumps(neigh)

def predict(a):
    lr_from_pickle = pickle.loads(saved_model)
    return(lr_from_pickle.predict_proba(a))