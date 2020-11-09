from main import *

def lin_reg_func_testerror(Dpar, features, targets, opt_lambda, Dtest):
    #Data is in textbook format global scope. Arguments are 
    # an index to train on, a list with features, a list with targets, optimal
    # lambda and an index to test on
    #Returns return genrelaization error when tested on Dtest
    #If multiple targets a vector with optimal errors is returned
    
    import numpy as np
    #!!! Sætte imports  hvor???
    
    # make a copy of data
    data_func_train = np.copy(data[Dpar,:])
    data_func_test = np.copy(data[Dtest,:])

    # Add offset attribute
    data_func_train = np.concatenate((np.ones((data_func_train.shape[0],1)),data_func_train),1)
    data_func_test = np.concatenate((np.ones((data_func_test.shape[0],1)),data_func_test),1)
    
    new_features = [x+1 for x in features] # because of added offset
    new_features = np.concatenate(([0],features)) # add offset to feature list   
    new_targets = [x+1 for x in targets]
    
    testerrors = np.empty(len(targets))
    t=0
    for new_target in new_targets: 
        y_train = data_func_train[:,new_target]
        X_train = data_func_train[:,new_features]
        y_test = data_func_test[:, new_target]
        X_test = data_func_test[:,new_features]
        N,M = X_train.shape # som indre loop i 2-level cross_val vil dette N svare til |Dpar|

        #function rlr_validate from toolbox_02450
        w = np.empty(M) #Vægtene.
                   
        # Standardize the training set and validation set based on training set moments (not the offset)
        mu = np.mean(X_train[:, 1:], 0)
        sigma = np.std(X_train[:, 1:], 0)
        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
            
        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
       

        # Compute parameters for current value of lambda and current CV fold
        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0,0] = 0 # remove bias regularization
        #Train model - get weights
        w = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Evaluate training and val performance
        train_error = np.power(y_train-X_train @ w.T,2).mean(axis=0)
        test_error = np.power(y_test-X_test @ w.T,2).mean(axis=0)
        testerrors[t] = test_error
        t+=1
    
    return testerrors
        



        
       
