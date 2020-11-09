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




#ANN initialize - Joel
vec_hidden_units = [1,2,3,4,5,6,7,8,9,10] # The range of hidden units to test


# -------- FUNCTIONS
def addToDict(dict, key, value):  # Adds a value to a speciffic key in a dict
    dict[key].append(value)

def removeFromDict(dict, key, value):
    if value in dict[key]:
        dict[key].remove(value)

def getVal(dict, key):
    return dict[key]

def minFromDict(dict):
    min = 100
    for key in range(len(dict)):

        if dict[key] < min:
            min = dict[key]


def ANNRegression(K, X, y, Dpar, s, vec_hidden_units):

    # Normalize
    #X = stats.zscore(Xd)
    #y = stats.zscore(yd)

    ANN_val_error = {}  # To store the validation error of each model

    # Setup a dict to store Error values for each hidden unit (key:map)
    for i in range(len(vec_hidden_units)):
        ANN_val_error[i] = []  # ANN_val_error = [[1, [error1, error2, error3]], [2 [error1, error2, error3]], ..., n]


    # Parameters for neural network classifier
    #n_hidden_units = 4      # number of hidden units
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000
    N, M = X.shape

    # K-fold crossvalidation
    #K = 3  # Number of folds (K2)
    #s = 10  # Number of models (I.e. Lambda and Hidden Unit values)
    iCV = model_selection.KFold(K, shuffle=True)

    # Inner fold
    for (k, (Dtrain, Dval)) in enumerate(iCV.split(X[Dpar, :], y[Dpar])):
        #print('\n   Inner fold: {0}/{1}'.format(k + 1, K))
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(stats.zscore(X[Dtrain, :]))
        y_train = torch.Tensor(stats.zscore(y[Dtrain]))
        X_test = torch.Tensor(stats.zscore(X[Dval, :]))
        y_test = torch.Tensor(stats.zscore(y[Dval]))

        for i in range(s):
            print(f"Inner: {k}, Model: {i}")
            # Define the model
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, vec_hidden_units[i]),  # M features to n_hidden_units
                torch.nn.Tanh(),  # 1st transfer function,
                torch.nn.Linear(vec_hidden_units[i], 1),  # n_hidden_units to 1 output neuron
                # no final tranfer function, i.e. "linear output"
            )
            loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

            errors = []  # make a list for storing generalizaition error in each loop

            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train,
                                                               y=y_train,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)

            #print('\n\tBest loss: {}\n'.format(final_loss))

            # Determine estimated class labels for test set
            y_test_est = net(X_test)

            # Determine errors and errors
            se = (y_test_est.float()-y_test.float())**2 # squared error
            mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
            errors.append(mse) # store error rate for current CV fold

            # Only store the new error if its better than the previous
            prevError = getVal(ANN_val_error, i)
            #print(f"Prev: {prevError}, MSE: {mse}")
            if not prevError: # Check whether there is an error
                addToDict(ANN_val_error, i, mse)
            elif mse < prevError:
                removeFromDict(ANN_val_error, i, prevError)
                addToDict(ANN_val_error, i, mse)


    # Find the best model from the CV folds
    # We do this by finding the model with lowest MSE
    bestError = getVal(ANN_val_error, 0)
    for i in range(len(ANN_val_error) - 1):
        if (getVal(ANN_val_error, i + 1) < bestError):
            #print(f"Best Error: {bestError}")
            removeFromDict(ANN_val_error, i, bestError)
            bestError = getVal(ANN_val_error, i + 1)

        elif getVal(ANN_val_error, i + 1):
            removeFromDict(ANN_val_error, i + 1, getVal(ANN_val_error, i + 1))

        else:
            addToDict(ANN_val_error, i, getVal(ANN_val_error, i))

    #print(ANN_val_error)

    for i in range(len(ANN_val_error)):
        val = getVal(ANN_val_error, i)
        if val:
            #print('Ran joel.py')
            return [i + 1, val] # Plus one because the n-hidden-units starts with 1
            #print([i,val])

    #print(ANN_val_error)
    #print('Ran joel.py')
    #return errors