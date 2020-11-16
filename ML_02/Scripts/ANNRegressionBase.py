# By Joel
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



# -------- FUNCTIONS

def ANN_regression_tester(Dpar, Dtest, X, y, n_hidden_units):
    # Parameters for neural network classifier
    #n_hidden_units = 4      # number of hidden units
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000
    N, M = X.shape

    # Nake a copy of data
    data_func_train = np.copy(data[Dpar, :])
    data_func_test = np.copy(data[Dtest, :])

    #print('Training model of type:\n\n{}\n'.format(str(model())))
    errors = [] # make a list for storing generalizaition error in each loop
    #for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):
    #print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))


    # Extract training and test set, convert to tensors
    X_train = torch.Tensor(stats.zscore(X[Dpar, :]))
    y_train = torch.Tensor(stats.zscore(y[Dpar]))
    X_test = torch.Tensor(stats.zscore(X[Dtest, :]))
    y_test = torch.Tensor(stats.zscore(y[Dtest]))
    N, M = X_train.shape

    # Define the model
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
    )

    #X_train = torch.Tensor(X_train)
    #y_train = torch.Tensor(y_train)
    #X_test = torch.Tensor(X_val)
    #y_test = torch.Tensor(y_val)


    loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)


    # Determine estimated class labels for test set
    y_test_est = net(X_test)

    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    errors.append(mse) # store error rate for current CV fold

    return mse