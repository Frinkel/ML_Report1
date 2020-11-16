import numpy as np
from sklearn import model_selection
from scipy import stats
from scipy.stats import norm
from lin_reg_func_testerror import *
from BM_model_test import *
from ANNRegressionBase import *
from main import *


features = [0,2,4,6,7,8]
target = [12]

X = data[:, features]
y = data[:, target]

N, M = X.shape
K = 10

oCV = model_selection.KFold(K, shuffle=True)

errors = dict()
for i in range(3):
    errors[i] = []
    
opt_lambda = 256 # Gen Error = 0.193
hidden_units = 1 # Gen Error = 0.237


# Arrays to store errors
lin_error = []
ANN_error = []
base_error = []



for (ok, (Dpar, Dtest)) in enumerate(oCV.split(X,y)):
    print(f"Fold: {ok}")
    #y_test = y[Dtest]


    # ANN vs Baseline
    #ANN_estimates = Stat_ANN_regression_tester(Dpar, Dtest, X, y, hidden_units)
    #print(ANN_estimates)
    # Determine errors and errors
    #y_test_ANN = torch.Tensor(y_test)
    #y_test_ANN = y_test_ANN.type(dtype=torch.uint8)
    #se = (ANN_estimates.float() - y_test_ANN.float()) ** 2  # squared error
    #mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean

    #print(mse)

    mse_ann = ANN_regression_tester(Dpar, Dtest, X, y, hidden_units)
    ANN_error.append(mse_ann[0])
    #print(mse_ann[0])

    mse_lin = lin_reg_func_testerror(Dpar, features, target, opt_lambda, Dtest)
    lin_error.append(mse_lin[0])
    #print(mse_lin[0])

    mse_base = bm_test_error(y, Dtest)
    base_error.append(mse_base)
    #print(mse_base)

print(f"Final Errors:")
print(ANN_error)
print(lin_error)
print(base_error)

print()
print("ANN vs. Base")
zAB = []
listAB = zip(ANN_error, base_error)
for i, j in listAB:
    zAB.append(i-j)

zhatAB = np.mean(ANN_error)-np.mean(base_error)
stdAB = np.std(zAB)
#cdfLAB = norm.cdf(zAB, zhatAB, stdAB)
#cdfUAB = norm.cdf(zAB, zhatAB, stdAB)

CIAB = stats.t.interval(alpha=0.95,df=len(zAB)-1,loc=zhatAB,scale=stdAB)
print(f"Confidence Interval: {CIAB}")
print(f"p val: {stats.ttest_rel(ANN_error, base_error)[1]}")
if stats.ttest_rel(ANN_error, base_error)[1] < 0.05:
    print("H0 rejeted")
else:
    print("H0 not rejected")

print()
print("Lin vs. Base")
zLB = []
listLB = zip(lin_error, base_error)
for i, j in listLB:
    zLB.append(i-j)

zhatLB = np.mean(lin_error)-np.mean(base_error)
stdLB = np.std(zLB)

CILB = stats.t.interval(alpha=0.95,df=len(zLB)-1,loc=zhatLB,scale=stdLB)
print(f"Confidence Interval: {CILB}")
print(f"p val: {stats.ttest_rel(lin_error, base_error)[1]}")
if stats.ttest_rel(lin_error, base_error)[1] < 0.05:
    print("H0 rejeted")
else:
    print("H0 not rejected")


print()
print("ANN vs. Lin")
zAL = []
listAL = zip(ANN_error, lin_error)
for i, j in listAL:
    zAL.append(i-j)

zhatAL = np.mean(lin_error)-np.mean(base_error)
stdAL = np.std(zAL)

CIAL = stats.t.interval(alpha=0.95,df=len(zAL)-1,loc=zhatAL,scale=stdAL)
print(f"Confidence Interval: {CIAL}")
print(f"p val: {stats.ttest_rel(ANN_error, lin_error)[1]}")
if stats.ttest_rel(ANN_error, lin_error)[1] < 0.05:
    print("H0 rejeted")
else:
    print("H0 not rejected")




