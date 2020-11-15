import matplotlib.pyplot as plt
import enum
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
from Logistisk_Regression import *
from Baseline_classifier import *
from ANN_Classification_max import *
from main import *
from toolbox_02450 import jeffrey_interval
from toolbox_02450 import mcnemar


features = [0,2,4,6,7,8]
target = [12]

X = data[:, features]
y = data[:, target]

N, M = X.shape

K = 10

CV = model_selection.KFold(K, shuffle=True)

yhat = []
y_true = []

#log_reg lambda value = 62.5055, 24.420530945486547

#ANN hidden units = 1
errors = dict()
for i in range(3):
    errors[i] = []
    
log_reg_param = 24.420530945486547

for (ok, (Dpar, Dtest)) in enumerate(oCV.split(X,y)):
    #print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    y_test = y[Dtest]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    dy = []
   
    y_est_log = train_test_model1(Dpar, Dtest, features, target, log_reg_param)
    dy.append( y_est_log )
    errors[0].append(np.sum(y_est_log != y_test[0]))
    #y_est_ANN =)
    #dy.append( y_est_ANN )
    #y_est_BASE =
    #dy.append( y_est_BASE )
   
    
    dy = np.stack(dy, axis=1)
    yhat.append(dy)
    y_true.append(y_test)

yhat = np.concatenate(yhat)
y_true = np.concatenate(y_true)
yhat[:,0] # predictions made by first classifier.
# Compute accuracy here.

# Compute the Jeffreys interval
alpha = 0.05
#[thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,0], alpha=alpha)
#print("Theta point estimate", thetahatA, " CI: ", CIA)

# Compute the mcnemar interval
alpha = 0.05
#[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)
#print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)