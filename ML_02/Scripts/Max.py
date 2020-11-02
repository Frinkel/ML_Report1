import numpy as np
import pandas as pd
import numpy as np

from main import *

X = data[:, [2,6,8]]

Y_EJ, Y_SC = data[:, 4], data[:, 7]

attributeNames = np.asarray(df.columns[range(0,13)])

N, M = X.shape

Test_EJ = np.empty(N)
Test_SC = np.empty(N)

def zero_rule_algorithm_regression(train, test):
	prediction = sum(train) / float(len(train))
	predicted = [prediction for i in range(len(test))]
	return predicted

predictions = zero_rule_algorithm_regression(Y_EJ, Test_EJ)

print(predictions)