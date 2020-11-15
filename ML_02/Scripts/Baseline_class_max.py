#Baseline classifier
from sklearn.dummy import DummyClassifier
from main import *

all_features = [0,2,4,6,8,12]
feature = [0,2,4,6,7,8]
target = [12]
X = data[:, feature]
y= data[:, target]

def bm_test(Dpar, Dtest):
    y_train = y[Dpar]
    X_train = X[Dpar,:]
    y_test = y[Dtest]
    X_test = X[Dtest,:]
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)  
    y_test_est = dummy_clf.predict(X_test)
    return y_test_est.tolist()
