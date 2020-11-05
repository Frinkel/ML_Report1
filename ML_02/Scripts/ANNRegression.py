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

def ANNRegression(X_train, y_train, X_val, y_val, n_hidden_units):
    # Parameters for neural network classifier
    #n_hidden_units = 4      # number of hidden units
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000
    N, M = X_train.shape

    # K-fold crossvalidation
    #K = 1                 # only three folds to speed up this example
    #CV = model_selection.KFold(K, shuffle=True)


    # Setup figure for display of learning curves and error rates in fold
    summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
    # Make a list for storing assigned color of learning curve for up to K=10
    color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
                  'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

    # ----------------------------------- Modificaiton
    y_test_est_models = []
    y_test_models = []

    #print('Training model of type:\n\n{}\n'.format(str(model())))
    errors = [] # make a list for storing generalizaition error in each loop
    #for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):
    #print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))

    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_val)
    y_test = torch.Tensor(y_val)

    # ----------------------------------- Modificaiton
    y_test_models.append(y_test)

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)

    print('\n\tBest loss: {}\n'.format(final_loss))

    # Determine estimated class labels for test set
    y_test_est = net(X_test)

    # ----------------------------------- Modificaiton
    y_test_est_models.append(y_test_est)

    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    errors.append(mse) # store error rate for current CV fold


    #print(f"Errors: {errors[1]}, y_test: {y_test_models}, y_test_est: {y_test_est_models}")
    # Find the best model from the CV folds
    # We do this by finding the model with lowest MSE
    errorsID = []
    for i in range(len(errors)):
        print(errors[i])
        errorsID.append([errors[i],i])

    #maxError = max(errors[0], errors[1], errors[2])
    bestError = []

    #for i in range(len(errorsID)):
    #    if(maxError == errorsID[i][0]):
    #        bestError = errorsID[i]
    #Largest MSE lowest loss?
    #print(bestError)


    print('Ran joel.py')
    return errors