import matplotlib.pyplot as plt
import enum
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
from Logistisk_Regression import *
from Baseline_class_max import *
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

#log_reg lambda value = 62.5055, 24.420530945486547, 5.963623316594637
#ANN hidden units = 1

errors = dict()
for i in range(3):
    errors[i] = []
    
log_reg_param = 24.420530945486547
hidden = 1

for (ok, (Dpar, Dtest)) in enumerate(oCV.split(X,y)):
    y_test = y[Dtest]

    dy = []
   
    y_est_log = train_test_model1(Dpar, Dtest, features, target, log_reg_param)
    dy.append( y_est_log )
    errors[0].append(np.sum(y_est_log != y_test[0]))
    
    y_est_ANN = ANNCFN(X, y, Dpar, Dtest, hidden)
    y_test_ANN = torch.Tensor(y_test)
    y_test_ANN = y_test_ANN.type(dtype=torch.uint8)
    e = (y_est_ANN != y_test_ANN)
    error_rate = ((sum(e).type(torch.float) / len(y_test_ANN))*100).data.numpy()
    err_rate = np.round(error_rate, 1)
    errors[1].append(err_rate[0])
    y_est_ANN = y_est_ANN.tolist()
    y_est_ANN = [x for sublist in y_est_ANN for x in sublist]
    dy.append( y_est_ANN )
    
    y_est_BASE = bm_test(Dpar, Dtest)
    dy.append( y_est_BASE )
    errors[2].append(np.sum(y_est_BASE != y_test[0]))
    
    dy = np.stack(dy, axis=1)
    yhat.append(dy)
    y_true.append(y_test)
    
yhat = np.concatenate(yhat)
yhat = yhat.astype(int)

y_true = np.concatenate(y_true).reshape(299)
y_true = y_true.astype(int)

print("\n")

# Compute the Jeffreys interval
alpha = 0.05
for j in range(3):
    [thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,j], alpha=alpha)
    print(f"{j}: Theta point estimate {thetahatA}, CI: {CIA} \n")

# Compute the McNemar test
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)
print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)