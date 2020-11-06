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
# exercise 8.2.6
# Load Matlab data file and extract variables of interest
#mat_data = loadmat('../../02450Toolbox_Python/Data/wine2.mat')
allAttributeNames = []
print(data.shape[1])
for x in range(data.shape[1]):
        allAttributeNames.append(Feature(x).name)
print(f"All attributes: \n {allAttributeNames} \n")

# Gather target feature and rest from the z-score standardized data
target_feature = [7]
predict_features = [2,6,8]

y = standData[:,target_feature]            # the target feature
X = standData[:,predict_features]        # the predict features
N, M = X.shape
C = 2

# Get the used attributenames
attributeNames = []
for i in range(len(predict_features)):
    attributeNames.append(Feature(predict_features[i]).name)
print(f"Input attributes: \n {attributeNames} \n")



## Normalize and compute PCA (change to True to experiment with PCA preprocessing)
do_pca_preprocessing = False
if do_pca_preprocessing:
    Y = stats.zscore(X,0)
    U,S,V = np.linalg.svd(Y,full_matrices=False)
    V = V.T
    #Components to be included as features
    k_pca = 3
    X = X @ V[:,:k_pca]
    N, M = X.shape



# Parameters for neural network classifier
n_hidden_units = 4      # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000

# K-fold crossvalidation
K = 3                  # only three folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)

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

print('Training model of type:\n\n{}\n'.format(str(model())))
errors = [] # make a list for storing generalizaition error in each loop
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))

    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index])

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

    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')

# Display the MSE across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
summaries_axes[1].set_xlabel('Fold')
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('MSE')
summaries_axes[1].set_title('Test mean-squared-error')


#print(f"Errors: {errors[1]}, y_test: {y_test_models}, y_test_est: {y_test_est_models}")
# Find the best model from the CV folds
# We do this by finding the model with (lowest MSE or largest??) (Lowest loss or largest??)
errorsID = []
for i in range(len(errors)):
    print(errors[i])
    errorsID.append([errors[i],i])

maxError = max(errors[0], errors[1], errors[2])
bestError = []

for i in range(len(errorsID)):
    if(maxError == errorsID[i][0]):
        bestError = errorsID[i]
#Largest MSE lowest loss?
print(bestError)


print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,2]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# Print the average classification error rate
print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))


# Get best model from folds
#for i in range(y_test_est.shape):
#print(y_test_models.shape)


# When dealing with regression outputs, a simple way of looking at the quality
# of predictions visually is by plotting the estimated value as a function of
# the true/known value - these values should all be along a straight line "y=x",
# and if the points are above the line, the model overestimates, whereas if the
# points are below the y=x line, then the model underestimates the value
plt.figure(figsize=(10,10))
y_est = y_test_est.data.numpy(); y_true = y_test.data.numpy()
axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('Alcohol content: estimated versus true value (for last CV-fold)')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()



plt.show()

print('Ran joel.py')