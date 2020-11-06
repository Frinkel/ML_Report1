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
#from joel.py import *

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

allAttributeNames = []
print(data.shape[1])
for x in range(data.shape[1]):
        allAttributeNames.append(Feature(x).name)
print(f"All attributes: \n {allAttributeNames} \n")

# Gather target feature and rest from the z-score standardized data
target_feature = [7]
predict_features = [2,4,6,8]

y = data[:,target_feature]            # the target feature
X = data[:,predict_features]        # the predict features

# Get the used attributenames
attributeNames = []
for i in range(len(predict_features)):
    attributeNames.append(Feature(predict_features[i]).name)
print(f"Input attributes: \n {attributeNames} \n")




#ANN init
vec_hidden_units = [1,2,3,4,5,6,7,8,9,10]









# Two level K-fold crossvalidation
oK = 10                 # Number of outer folds
iK = 10                 # Number of inner folds
oCV = model_selection.KFold(oK, shuffle=True)
iCV = model_selection.KFold(iK, shuffle=True)

# Outer fold
for (ok, (outer_train_index, outer_test_index)) in enumerate(oCV.split(X,y)):
    print('\nCrossvalidation outer fold: {0}/{1}'.format(ok + 1, oK))
    #print(outer_train_index)
    #print("\n")
    #print(outer_test_index)

    #print(X[outer_train_index, :]) # the predict
    #print("\n")
    #print(y[outer_train_index]) # the target



    #for i in range(10):
    #    print(i)
    #    ANNRegression(X[outer_train_index, :], y[outer_train_index], vec_hidden_units[i])

    # Inner fold
    for (ik, (inner_train_index, inner_test_index)) in enumerate(iCV.split(X[outer_train_index, :],y[outer_train_index])):
        print('\n   Crossvalidation inner fold: {0}/{1}'.format(ik + 1, iK))
        #print(X[inner_train_index, :].shape) # the predict
        #print("\n")
        #print(len(y[inner_train_index])) # the target

        #Train each model
        #ANN

        #Linear regression

        #Base model




print('Ran two-level-cv.py')