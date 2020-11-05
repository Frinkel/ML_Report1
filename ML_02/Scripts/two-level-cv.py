# Import packages
import matplotlib.pyplot as plt
from main import *
import enum
import matplotlib.pyplot as plt
import numpy as np
#from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
from ANNRegression import *

# Data contains 13 features and 299 observations

## Features:
# 0 = age                       - int
# 1 = anaemia                   - boolean
# 2 = creatine phosphokinase    - int
# 3 = diabetes                  - boolean
# 4 = ejection fraction         - percentage
# 5 = high blood pressure       - bool
# 6 = platelets                 - int/float
# 7 = serum creatine            - float
# 8 = serum sodium              - int
# 9 = sex                       - binary
# 10 = smoking                  - boolean
# 11 = time                     - int
# 12 = death event              - boolean

# Enumerate the different features
from main import data

# Create enumeration for easier access to values and names
class Feature(enum.Enum):
    age = 0
    anaemia = 1
    CPK = 2
    diabetes = 3
    eject_fraction = 4
    HBP = 5
    platelets = 6
    ser_creatinine = 7
    ser_sodium = 8
    sex = 9
    smoking = 10
    time = 11
    death = 12



# -------- FUNCTIONS

# Get Data
allAttributeNames = []
print(data.shape[1])
for x in range(data.shape[1]):
        allAttributeNames.append(Feature(x).name)
print(f"All attributes: \n {allAttributeNames} \n")

# Gather target feature and rest from the z-score standardized data
target_feature = [7]
predict_features = [2,4,6,8]

y = data[:,target_feature]          # the target feature
X = data[:,predict_features]        # the predict features

# Get the used attributenames
attributeNames = []
for i in range(len(predict_features)):
    attributeNames.append(Feature(predict_features[i]).name)
print(f"Input attributes: \n {attributeNames} \n")




#ANN initialize - Joel
vec_hidden_units = [1,2,3,4,5,6,7,8,9,10] # The range of hidden units to test
ANN_val_error = {} # To store the validation error of each model

# Setup a dict to store Error values for each hidden unit (key:map)
for i in range(len(vec_hidden_units)):
    ANN_val_error[i] = []   # ANN_val_error = [[1, [error1, error2, error3]], [2 [error1, error2, error3]], ..., n]

def addToDict(dict, key, value):  # Adds a value to a speciffic key in a dict
    dict[key].append(value)

def minFromDict(dict):
    min = 100
    for key in range(len(dict)):

        if dict[key] < min:
            min = dict[key]



# Two level K1-, K2-fold crossvalidation
oK = 10                 # Number of outer folds (K1)
iK = 10                 # Number of inner folds (K2)
s = 10                  # Number of models (I.e. Lambda and Hidden Unit values)

oCV = model_selection.KFold(oK, shuffle=True)
iCV = model_selection.KFold(iK, shuffle=True)

# Outer fold
for (ok, (Dpar, Dtrain)) in enumerate(oCV.split(X,y)):
    print('\nOuter fold: {0}/{1}'.format(ok + 1, oK))

    # Inner fold
    for (ik, (Dtrain, Dval)) in enumerate(iCV.split(X[Dpar, :],y[Dpar])):
        print('\n   Inner fold: {0}/{1}'.format(ik + 1, iK))

        # Gather the training and validation data
        X_train = X[Dtrain, :]
        y_train = y[Dtrain]
        X_val = X[Dval, :]
        y_val = y[Dval]

        for i in range(s):
            #print(i)

            # Train the models
            ANNError = ANNRegression(X_train, y_train, X_val, y_val, vec_hidden_units[i])
            #ANNError = ANNRegression(X_train, y_train, X_val, y_val, vec_hidden_units[i]) # Returns the validation error

            addToDict(ANN_val_error, i, ANNError)

    print(ANN_val_error)
    quit(100)

print('Ran two-level-cv.py')