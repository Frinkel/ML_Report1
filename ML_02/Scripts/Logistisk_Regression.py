import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.linear_model as lm
from sklearn import model_selection
from main import *

all_features = [2,4,6,8,12]
feature = [2,4,6,8]
target = [12]
X = data[:, feature]
y= data[:, target]

oK = 10
oCV = model_selection.KFold(oK, shuffle=True)

result_lambda = np.empty(oK)

def log_reg_func(Dpar, features, targets):
    data_func = np.copy(data[Dpar,:])
    y = data_func[:,targets]
    X = data_func[:,features]

    N, M = X.shape

    K = 10
    CV = model_selection.KFold(K, shuffle=True)
    
    f = 0
    y = y.squeeze()
    fold_size = np.empty(K)

    lambda_interval = np.logspace(-8, 2, 50)
    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    #coefficient_norm = np.zeros(len(lambda_interval))
    genErrors = dict()
    
    for i in range(0, len(lambda_interval)):
        genErrors[i] = []
    
    for train_index, val_index in CV.split(X,y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]
        fold_size[f] = len(val_index)
        #print(fold_size[f])
        
        mu = np.mean(X_train, 0)
        sigma = np.std(X_train, 0)
        X_train = (X_train - mu) / sigma
        X_val = (X_val - mu) / sigma
            
        for k in range(0, len(lambda_interval)):
            mdl = lm.LogisticRegression(penalty='l2', C=1/lambda_interval[k])

            mdl.fit(X_train, y_train)

            y_train_est = mdl.predict(X_train).T
            y_test_est = mdl.predict(X_val).T
                
            train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
            genErrors[k].append(np.sum(y_test_est != y_val) / len(y_val))
                
    f += 1
    #print(genErrors)
    for n in range(0, len(lambda_interval)):
        arr = genErrors.get(n)
        for y in range(0, K):
            test_error_rate[n] += (arr[y] * fold_size[y] / N)
            
    #print(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    min_error = test_error_rate[opt_lambda_idx]
    opt_lambda = lambda_interval[opt_lambda_idx]
    #print(min_error)
    
    return opt_lambda

def train_test_model(Dpar, Dtest, features, targets, reg_param):
    data_func_train = np.copy(data[Dpar,:])
    data_func_test = np.copy(data[Dtest,:])
    
    X_train = data_func_train[:, features]
    y_train = data_func_train[:, targets]
    X_test = data_func_test[:, features]
    y_test = data_func_test[:, targets]
    
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    
    mdl = lm.LogisticRegression(penalty='l2', C=reg_param)

    mdl.fit(X_train, y_train.ravel())

    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T
                
    train_error_rate = np.sum(y_train_est != y_train) / len(y_train)
    t_error_rate = np.sum(y_test_est != y_test) / len(y_test)
    
    #y_est_dead_prob = mdl.predict_proba(X_test)[:, 1]
    
    #y_est_alive_prob = mdl.predict_proba(X_test)[:, 0]
    
    #x_class = mdl.predict_proba(X_test)[0,1]
    #print("X_class", x_class)
    
    return t_error_rate

for (ok, (Dpar, Dtest)) in enumerate(oCV.split(X,y)):
    result_lambda[ok] = log_reg_func(Dpar, feature, target)
    print("t_error_rate", train_test_model(Dpar, Dtest, feature, target, result_lambda[ok]))