#Baseline classifier

from sklearn.dummy import DummyClassifier


def bm_test_error(Dpar, Dtest):
    y_train = y[Dpar]
    X_train = X[Dpar,:]
    y_test = y[Dtest]
    X_test = X[Dtest,:]
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)  
    return 1-dummy_clf.score(X_test, y_test)