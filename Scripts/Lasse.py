# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from main import *
from enum import Enum
from scipy.linalg import svd


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
class feature:
    age = 0
    anaemia = 1
    creatine = 2
    diabetes = 3
    eject_fract = 4
    HBP = 5
    platelets = 6
    ser_creatine = 7
    ser_sodium = 8
    sex = 9
    smoking = 10
    time = 11
    death = 12


#--Principal Component Analysis - PCA ---(script 2.1.1-2.1.6)-----------------
    #OBS!!: Time continuous?
    # Præcis definition af time?: Hvor mange dage der er gået før der blev fulgt op på patienten?
    #Droppe Død/levende fordi vi vil bruge det til klassificering?
    # Se bort fra tid? Tid til der blev tjekket... virker blot forstyrrende for klassificering hvis vi ikke har statistiske muskler til at analysere betinget på tiden.
    # Binære features i PCA???


#Sæt data op som i bogen: X, N, M, y, C
X = data
y = X[:,-1] 
X = X[:,0:12] #OBS!!! Excluded last coloumn since we use it to classify against (also exclude time?...fordi... bineære features???)
N,M = X.shape
attributeNames = np.asarray(df.columns[0:13])

#Klassificere som død/levende:
classNames = np.array(["Dead", "Not dead"])
C=len(classNames)



# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y = X - np.ones((N, 1))*X.mean(0)
Y = Y*(1/np.std(Y,0))


# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T  

#_---------------------------------2.1.3
# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 
g=S*S
threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

#----2.2.2--
figure()
plt.plot(range(1,len(rho)+1),rho,'o-')
title('Variance explained by principal components');
xlabel('Principal component');
ylabel('Variance explained value');
#--------------------------------------2.1.4 C=2 (død/levende)
# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('Heartfailure data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()

#----------------------------------------2.1.5
# A look at the coefficients of the first 3 principal components:
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, r-1) #så attributnummer passer med AttributeNames
plt.xlabel('Attributes by number')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Heartfailure: PCA Component Coefficients')
plt.show()

# print("It seems that PC2 sort of divides 'dead' and 'not dead' patients. Inspection of the second principal component:")
# print('PC2:')
# print(V[:,1].T)

# print('Consider the first observation:') # in normalized data
# print(Y[0,:])
# print('...and its projection onto PC2')
# print(Y[0,:]@V[:,1])

#--------------------------------------2.1.6
# Plot attribute coefficients in principal component space
# Indices of the principal components to be plotted against
i = 0
j = 1
plt.figure()
for att in range(V.shape[1]):
    plt.arrow(0,0, V[att,i], V[att,j])
    #plt.text(V[att,i], V[att,j], attributeNames[att]) OBS! ikke pænt med navne oven i hinanden. Næste linje er dog ikke meget pænere.
    plt.text(V[att,i], V[att,j], att)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel('PC'+str(i+1))
plt.ylabel('PC'+str(j+1))
plt.grid()
# Add a unit circle
plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), np.sin(np.arange(0, 2*np.pi, 0.01)));
plt.title('Attribute coefficients')
plt.axis('equal')

#-----------------------Ekstra plots-------------------------
# Indices of the principal components to be plotted
i = 0
j = 2

# Plot PCA of the data
f = figure()
title('Heartfailure data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()

# Indices of the principal components to be plotted
i = 1
j = 2

# Plot PCA of the data
f = figure()
title('Heartfailure data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()
