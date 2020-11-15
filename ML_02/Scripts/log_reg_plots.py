import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.linear_model as lm
from sklearn import model_selection
from main import *

feature = [0,2,4,6,7,8]
target = [12]
X = data[:, feature]
y = data[:, target]
N, M = X.shape  
K = 10

result_lambda = np.empty(K)

CV = model_selection.KFold(K, shuffle=True)
    
f = 0
y = y.squeeze()
fold_size = np.empty(K)
lambda_interval = np.logspace(-8, 2, 50)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
#coefficient_norm = np.zeros(len(lambda_interval))
genErrors = dict()
trainErrors = dict()
    
for i in range(0, len(lambda_interval)):
    genErrors[i] = []
    trainErrors[i] = []
    
for train_index, val_index in CV.split(X,y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_val = X[val_index]
    y_val = y[val_index]
    fold_size[f] = len(val_index)
        
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma
            
    for k in range(0, len(lambda_interval)):
        mdl = lm.LogisticRegression(penalty='l2', C=1/lambda_interval[k])

        mdl.fit(X_train, y_train)

        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_val).T
                
        trainErrors[k].append(np.sum(y_train_est != y_train) / len(y_train))
        genErrors[k].append(np.sum(y_test_est != y_val) / len(y_val))
        
f += 1

for n in range(0, len(lambda_interval)):
    arr = genErrors.get(n)
    arrr = trainErrors.get(n)
    for y in range(0, K):
        test_error_rate[n] += (arr[y] * fold_size[y] / N)
        train_error_rate[n] += (arrr[y] * fold_size[y] / N)

#print(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
min_error = test_error_rate[opt_lambda_idx]
opt_lambda = lambda_interval[opt_lambda_idx]
#print(min_error)
    
plt.figure(figsize=(8,8))
#plt.plot(np.log10(lambda_interval), train_error_rate*100)
#plt.plot(np.log10(lambda_interval), test_error_rate*100)
#plt.plot(np.log10(opt_lambda), min_error*100, 'o')
plt.semilogx(lambda_interval, train_error_rate*100)
plt.semilogx(lambda_interval, test_error_rate*100)
plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
plt.ylim([0, 4])
plt.grid()
plt.show()