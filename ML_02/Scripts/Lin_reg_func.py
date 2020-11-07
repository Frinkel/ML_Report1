def lin_reg_func(Dpar, features, targets):
    #Data is in textbook format global scope. Arguments are
    # an index of observations, a list with features and a list with targets
    #Returns optimal regularization factor lambda
    #If multiple targets a vector with optimal lambdas is returned
    
    from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid)
    import numpy as np
    import sklearn.linear_model as lm
    from sklearn import model_selection
    from toolbox_02450 import rlr_validate
    #!!! Sætte imports hvor???
    
    # make a copy of data
    data_func = np.copy(data[Dpar,:])
    # Add offset attribute
    data_func = np.concatenate((np.ones((data_func.shape[0],1)),data_func),1)
    
    new_features = [x+1 for x in features] # because of added offset
    new_features = np.concatenate(([0],features)) # add offset to feature list   
    new_targets = [x+1 for x in targets]

    cvf = 10 # 10-fold cross-validation
    K = cvf
    
    # Values of lambda
    #lambdas = np.power(10.,range(-5,9)) 
    lambdas = np.power(2., range(-10,20)) #!!! omkring 10^2 (2^7) er interessant
    
    result = np.empty(len(targets))
    t=0
    for new_target in new_targets: 
        y = data_func[:,new_target]
        X = data_func[:,new_features] 
        N,M = X.shape # som indre loop i 2-level cross_val vil dette N svare til |Dpar|
        
        #function rlr_validate from toolbox_02450
        CV = model_selection.KFold(cvf, shuffle=True)
        w = np.empty((M,cvf,len(lambdas))) #Vægtene. Dimensioner: features x folds x lambdas
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
        
        result[t] = opt_lambda
        t+=1
        
    return result    
        



        
       
