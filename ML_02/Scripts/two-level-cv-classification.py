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
from Logistisk_Regression import *
from Baseline_classifier import *
from ANNBinaryCalssification import *

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
#print(f"All attributes: \n {allAttributeNames} \n")

# Gather target feature and rest from the z-score standardized data
target_feature = [12]
predict_features = [0,2,4,6,7,8]

y = data[:,target_feature]          # the target feature
X = data[:,predict_features]        # the predict features

# Get the used attributenames
attributeNames = []
for i in range(len(predict_features)):
    attributeNames.append(Feature(predict_features[i]).name)
#print(f"Input attributes: \n {attributeNames} \n")

# Define the range of hidden units should be tested
vec_hidden_units = [1,2,3,4,5,6]

# Two level K1-, K2-fold crossvalidation
oK = 2                # Number of outer folds (K1)
iK = 10               # Number of inner fold (K2) only used by ANN

oCV = model_selection.KFold(oK, shuffle=True)
iCV = model_selection.KFold(iK, shuffle=True)

# Create an array to store linreg errors
log_testerror = np.empty((oK, 1))

# Create an array to store baseline errors
base_testerror = []

# Create a dict to store the ANN errors
ANN_gen_error = []

# Outer fold
for (ok, (Dpar, Dtest)) in enumerate(oCV.split(X,y)):
    print('\nOuter fold: {0}/{1}'.format(ok + 1, oK))

    # Train models on Dpar
        # Return best model trained
    print("* Training Models *")
    
    # Lin Reg model
    #opt_lambda = log_reg_func(Dpar, predict_features, target_feature)

    # ANN model
    ANNBestModel = ANNClassification(iK, X, y, Dpar, len(vec_hidden_units), vec_hidden_units)
    print(f"ANN best model found = {ANNBestModel[0]} hidden units.")


    # Test best model on Dtest
        # Return Error
    print("* Testing Best Model *")
    # Lin Reg model

    log_testerror[ok] = train_test_model(Dpar, Dtest, predict_features, target_feature, opt_lambda)
    print(f"Log Reg Generalisation error = {log_testerror[ok]}.")

    # ANN model
    ANNGenError = ANNClassification(iK, X, y, Dtest, 1, [ANNBestModel[0]])
    print(f"ANN Generalisation error = {ANNGenError[1][0][0]} with {ANNBestModel[0]} hidden units.")
    ANN_gen_error.append([ANNBestModel[0], ANNGenError[1][0][0]])  # [Opt model, Gen error]

    # Basic model
    base_test = bm_test_error(Dpar, Dtest)
    print(f"Base model generalisation error = {base_test}")
    base_testerror.append(base_test)

    # Exit after first outer fold
    #quit(100)

print("")
print("Final errors:")
print(f"All ANN errors: {ANN_gen_error}")
print(f"All Base errors: {base_testerror}")

print('Ran two-level-cv.py')