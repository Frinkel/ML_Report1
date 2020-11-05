import pandas as pd
import numpy as np
from sklearn import model_selection

from main import *

X = data[:, [2,6,8]]

Y_EJ, Y_SC = data[:, 4], data[:, 7]

attributeNames = np.asarray(df.columns[range(0,13)])

N, M = X.shape

Test_EJ = np.empty(N)
Test_SC = np.empty(N)

def zero_rule_algorithm_regression(train, test):
	prediction = np.mean(train, axis=0)
	predicted = np.asarray([prediction for i in range(len(test))])
	return predicted

predictions = zero_rule_algorithm_regression(Y_EJ, Test_EJ)

print('Predictions'.format(predictions))

def cross_fold_algorithm(X, y):
        # K-fold crossvalidation
        K = 10
        CV = model_selection.KFold(n_splits=K,shuffle=True)
        
        Error_train = np.empty((K,1))
        Error_test = np.empty((K,1))
        
        k=0
        for train_index, test_index in CV.split(X):
            
            # extract training and test set for current CV fold
            X_train = X[train_index,:]
            y_train = y[train_index]
            X_test = X[test_index,:]
            y_test = y[test_index]
            
            # Compute squared error
            Error_train[k] = np.square(y_train-predictions.sum()/len(y_train))
            Error_test[k] = np.square(y_test-predictions.sum()/len(y_train))
                                      
            print('Cross validation fold {0}/{1}'.format(k+1,K))
            print('Train indices: {0}'.format(train_index))
            print('Test indices: {0}'.format(test_index))
            
            k+=1
            
cross_fold_algorithm(X, Y_EJ)