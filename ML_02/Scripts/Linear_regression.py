# Search for optimal regularization parameter for linear regression model
from main import *

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid)
import numpy as np
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate


# Add offset attribute and set attributeNames
data = np.concatenate((np.ones((data.shape[0],1)),data),1)
attributeNames = np.asarray(df.columns[0:13])
attributeNames = np.append(['Offset'], attributeNames)

#Sæt data op som i bogen: X, N, M, y
features = [0,1,2,3,4,6,7,9,10,11] # Husk offset(0). Vi ville predicte ud fra features Creatinine phosphate(3), plattelets(7) and Serum sodium(9).
targets = [5,8] # Already added offset, so ejection fratction is 5 and serum creatinine 8
cvf = 10 # 10-fold cross-validation
K = cvf

# Values of lambda
#lambdas = np.power(10.,range(-5,9)) 
lambdas = np.power(2., range(-10,20)) #!!! omkring 10^2 er interessant

for target in targets: 
    y = data[:,target]
    X = data[:,features] 
    N,M = X.shape # som indre loop i 2-level cross_val vil dette N svare til |Dpar|
    
    #function rlr_validate from toolbox_02450
    CV = model_selection.KFold(cvf, shuffle=True)
    w = np.empty((M,cvf,len(lambdas)))
    train_error = np.empty((cvf,len(lambdas)))
    val_error = np.empty((cvf,len(lambdas)))
    f = 0
    y = y.squeeze()
    fold_size = np.empty(K)
    
    for train_index, val_index in CV.split(X,y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]
        fold_size[f] = len(val_index) # |D-val| om man vil
        
        # Standardize the training set and validation set based on training set moments (not the offset)
        mu = np.mean(X_train[:, 1:], 0)
        sigma = np.std(X_train[:, 1:], 0)
        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_val[:, 1:] = (X_val[:, 1:] - mu) / sigma
        
        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
   
        for l in range(0,len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0,0] = 0 # remove bias regularization
            #Train model - get weights
            w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            # Evaluate training and val performance
            train_error[f,l] = np.power(y_train-X_train @ w[:,f,l].T,2).mean(axis=0)
            val_error[f,l] = np.power(y_val-X_val @ w[:,f,l].T,2).mean(axis=0)
    
        f=f+1 
        
    
    #Generalization errors for different values of lambda (algorithm 5)
    Gen_error = np.zeros(len(lambdas))
    Train_error = np.zeros(len(lambdas))
    for s in range(len(lambdas)):
        for k in range(0,K):
            Gen_error[s] += val_error[k][s]*fold_size[k]/N   #fold_size[k] er |D_val| i fold k N er størrelsen på det dataset funktionen fodres med, dvs |Dpar|    
            Train_error[s]+= train_error[k][s]*fold_size[k]/N
    
    #Optimal model (optimal regularization value)
    opt_lambda = lambdas[np.argmin(Gen_error)]
    opt_lambda_idx = np.argmin(Gen_error)
    
    
    show()
    # Display results
    print('\n Regularized linear regression for predicting {0}:'.format(attributeNames[target]))
    print('Optimal regularization constant(lambda) suggested by {0}-fold crossvalidation is around {1}'.format(K, opt_lambda))
    print('The generalization error for this optimal model in this inner loop was {0}. \n NOTE: this is ONLY 1-level cross-validation.'.format(Gen_error[opt_lambda_idx]))

    print('\n The means of the weights over the 10 folds using the optimal model were:')
    for feature in range(M):
        #print(' la la {0}'.format(attributeNames[features[feature]))
        print('The weight of attribute {0} was {1} '.format(attributeNames[features[feature]], np.mean(w[feature,:,opt_lambda_idx])))
        #print('The weight of attribute {0} was {1}'.format(attributeNames[features[feature]], np.mean(w[feature,:,opt_lambda_idx], axis=1)))
        #!!!!!! hvorfor virker dette ikke??!?
    
    
    figure(figsize=(10,10)) 
    title('Optimal lambda: {0}'.format(opt_lambda))
    semilogx(lambdas,Train_error,'b.-', lambdas, Gen_error, 'r.-')
    xlabel('Regularization factor')
    ylabel('Mean squared error (crossvalidation)')
    legend(['Weighted averaged train error','Generalization error'])
    grid()
    show()  


    figure(figsize=(12,12))
    semilogx(lambdas,np.mean(w[1:,:,:], axis=1).T,'.-') # Don't plot the bias term
    xlabel('Regularization factor')
    ylabel('Mean Coefficient Values')
    grid()
    legend(attributeNames[1:], loc='best') #!!! omit this
    show()
    
    
    
    #!!!Slet alt herunder
    
    figure(figsize=(10,10))
    semilogx(lambdas, Gen_error,'r.-')
    ylabel('Generalization error')
    xlabel('Regularization strength, $\log_{10}(\lambda)$')
    title('Estimated generalization error as a function of $\lambda$')
    grid()
    show()
    
    
    # figure(figsize=(8,8)) 
    # title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
    # loglog(lambdas,Train_error,'b.-')
    # xlabel('Regularization factor')
    # ylabel('Squared error (crossvalidation)')
    # legend(['Train error','Validation error'])
    # grid()
    # show()      
    

        
       
