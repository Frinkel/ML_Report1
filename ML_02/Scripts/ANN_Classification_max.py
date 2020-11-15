# Import packages
import matplotlib.pyplot as plt
import enum
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary
from scipy import stats
    
def ANNCFN(X, y, Dtrain, Dtest, s, hidden_units):
    # Normalize
    # X = stats.zscore(Xd)
    # y = stats.zscore(yd)

    # Parameters for neural network classifier
    n_replicates = 1  # number of networks trained in each k-fold
    max_iter = 10000
    #max_iter = 1000
    N, M = X.shape

    # K-fold crossvalidation
    # K = 3  # Number of folds (K2)
    # s = 10  # Number of models (I.e. Lambda and Hidden Unit values)

    X_train = torch.Tensor(stats.zscore(X[Dtrain, :]))
    y_train = torch.Tensor(stats.zscore(y[Dtrain]))
    X_test = torch.Tensor(stats.zscore(X[Dtest, :]))
    y_test = torch.Tensor(stats.zscore(y[Dtest]))


    # The lambda-syntax defines an anonymous function, which is used here to
    # make it easy to make new networks within each cross validation fold
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, vec_hidden_units[i]),  # M features to H hiden units
                # 1st transfer function, either Tanh or ReLU:
        torch.nn.Tanh(),  # torch.nn.ReLU(),
        torch.nn.Linear(vec_hidden_units[i], 1),  # H hidden units to 1 output neuron
        torch.nn.Sigmoid()  # final tranfer function
    )

    loss_fn = torch.nn.BCELoss()

    errors = [] # make a list for storing generalizaition error in each loop

    net, final_loss, learning_curve = train_neural_net(model,
                                                             loss_fn,
                                                             X=X_train,
                                                             y=y_train,
                                                             n_replicates=n_replicates,
                                                             max_iter=max_iter)

    # Determine estimated class labels for test set
    y_sigmoid = net(X_test)  # activation of final note, i.e. prediction of network
    y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8)  # threshold output of sigmoidal function
    y_test = y_test.type(dtype=torch.uint8)
    # Determine errors and error rate
    e = (y_test_est != y_test)
    error_rate = (sum(e).type(torch.float) / len(y_test)).data.numpy()
    errors.append(error_rate)  # store error rate for current CV fold
    
    return error_rate