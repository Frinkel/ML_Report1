import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import sklearn.linear_model as lm
from sklearn import model_selection
from scipy import stats
from main import *
sns.set(style="white")

features = [0,2,4,6,7,8]
target = [12]

X = data[:, features]
y = data[:, target]

N, M = X.shape

K = 10

CV = model_selection.KFold(K, shuffle=True, random_state=1)

reg_param = 1e-8

avgWeights = []

for (k, (Dpar, Dtest)) in enumerate(CV.split(X,y)):
    
    X_train = X[Dpar, :]
    y_train = y[Dpar, :].ravel()
    X_test = X[Dtest, :]
    y_test = y[Dtest, :].ravel()
    
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    
    mdl = lm.LogisticRegression(penalty='l2', C=1/reg_param)

    mdl.fit(X_train, y_train)

    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T
                
    train_error_rate = np.sum(y_train_est != y_train) / len(y_train)
    t_error_rate = np.sum(y_test_est != y_test) / len(y_test)
    
    b = mdl.intercept_[0]
    w_est = mdl.coef_[0]
    avgWeights.append(w_est.ravel())
    coefficient_norm = np.sqrt(np.sum(w_est**2))

avgW = np.asarray(avgWeights)
avgW = avgW.sum(axis=0) / 10


