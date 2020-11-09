import pandas as pd
import numpy as np
from sklearn import model_selection

from main import *

X = data[:, [2,4,6,8]]

Y_SC = data[:, 7]

#attributeNames = np.asarray(df.columns[range(0,13)])

N, M = X.shape

Test_SC = np.empty(N)

def zero_rule_algorithm_regression(train, test):
	prediction = np.mean(train, axis=0)
	predicted = np.asarray([prediction for i in range(len(test))])
	return predicted

predictions = zero_rule_algorithm_regression(Y_SC, Test_SC)

def cross_fold_algorithm(X, y):
   # K-fold crossvalidation
   K = 10
   k = 0
   CV = model_selection.KFold(n_splits=K,shuffle=True)
       
   Error_test = np.empty((K,1))
        
   for train_index, test_index in CV.split(X):
       
       # Extract test set for current CV fold
       y_test = y[test_index]
       p_test = predictions[test_index]
            
       # Compute squared error
       Error_test[k] = np.square(y_test-p_test).sum()/y_test.shape[0]
                                      
       k += 1
       
   #print('Error Test', Error_test)
   
   return min(Error_test)

Err_test = cross_fold_algorithm(X, Y_SC)