import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from main import *


def lin_reg_func(Dpar, features, targets):
    data_func = np.copy(data[Dpar,:])
    y = data_func[:,targets]
    X = data_func[:,features]
    X = data[:, [0,2,4,6,7,8]]
    y = data[:,12]

attributeNames = np.asarray(df.columns[range(0,12)])
classNames = np.asarray(df.columns[range(12,13)])
c = len(classNames)

N, M = X.shape

K = 10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.95, stratify=y)

mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

lambda_interval = np.logspace(-8, 2, 50)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))

for k in range(0, len(lambda_interval)):
    mdl = lm.LogisticRegression(penalty='l2', C=1/lambda_interval[k])

    mdl.fit(X_train, y_train)

    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T
    
    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)
    
    #w_est = mdl.coef_[0]
    #coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
    
    y_est_dead_prob = mdl.predict_proba(X_test)[:, 1]
    y_est_alive_prob = mdl.predict_proba(X_test)[:, 0]

    x_class = mdl.predict_proba(X_test)[0,1]
    
    #print('\nProbability of given patient being dead: {0:.4f}'.format(x_class))
    print('\nOverall misclassification rate: {0:.3f}'.format(test_error_rate[k]))
    
opt_lambda_idx = np.argmin(test_error_rate)
min_error = test_error_rate[opt_lambda_idx]
opt_lambda = lambda_interval[opt_lambda_idx] 