import pandas as pd
import numpy as np
from sklearn import model_selection
from main import *

def zero_rule_algorithm_regression(train, test):
	prediction = np.mean(train, axis=0)
	predicted = np.asarray([prediction for i in range(len(test))])
	return predicted

def predict(y):
    N = y.shape
    test = np.empty(N)
    predictions = zero_rule_algorithm_regression(y, test)
    return predictions
    
def bm_test_error(y, test_index):
    mean_predicted = predict(y)
    #print("predictions ", mean_predicted)
    
    y_test = y[test_index]
    #print("y test ", y_test)
    p_test = mean_predicted[test_index]
    #print("p test ", p_test)
            
    # Compute squared error
    Error_test = np.square(y_test-p_test).sum()/y_test.shape[0]
    return Error_test