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
from Lin_reg_func import *
from lin_reg_func_testerror import *
from Max import *

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


# Define the range of hidden units should be tested
vec_hidden_units = [1,2,3,4,5,6,7,8,9,10]


# Two level K1-, K2-fold crossvalidation
oK = 3                 # Number of outer folds (K1)
iK = 3                 # Number of inner folds (K2)

oCV = model_selection.KFold(oK, shuffle=True)
iCV = model_selection.KFold(iK, shuffle=True)

# Create an array to store linreg errors
lin_testerror = np.empty(oK)

# Outer fold
for (ok, (Dpar, Dtest)) in enumerate(oCV.split(X,y)):
    print('\nOuter fold: {0}/{1}'.format(ok + 1, oK))

    # Train models on Dpar
        # Return best model trained
    print("* Training Models *")
    # Basic model
    # cross_fold_algorithm(Dpar, Dtest)

    # Lin Reg model
    opt_lambda = lin_reg_func(Dpar, predict_features, target_feature)
    lin_testerror[oK] = lin_reg_func_testerror(Dpar, predict_features, target_feature, opt_lambda, Dtest)

    # ANN model
    ANNBestModel = ANNRegression(iK, X, y, Dpar, len(vec_hidden_units), vec_hidden_units)
    print(f"ANN best model found = {ANNBestModel[0]} hidden units.")




    # Test best model on Dtest
        # Return Error
    print("* Testing Best Model *")
    # Lin Reg model
    lin_test = lin_reg_func_testerror(Dtest, predict_features, target_feature, opt_lambda, Dtest)
    print(f"ANN Generalisation error = {lin_test[1]}.")
    # ANN model
    ANNGenError = ANNRegression(iK, X, y, Dtest, 1, [ANNBestModel[0]])
    print(f"ANN Generalisation error = {ANNGenError[1]} with {ANNBestModel[0]} hidden units.")

    #Exit after first outer fold
    quit(100)

print('Ran two-level-cv.py')