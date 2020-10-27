# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import *
from enum import Enum


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
 
    
# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the columns from inspecting 
# the file.
cols = range(0, 13) 
X = data[:, cols]
K = np.empty((299, 1))

#p = df.describe()

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])

N, M = X.shape

summary_Array = []
arr = []
j = 0

# Compute values
for i in range(1,12):
    #0 Mean
    arr.append(data[:,i].mean())
    
    #1 STD
    arr.append(data[:,i].std(ddof=1))
    
    #2 Median
    arr.append(np.median(data[:,i]))
    
    #3 Var
    arr.append(np.var(data[:,i]))
    
    #4 Range
    arr.append(data[:,i].max()-data[:,i].min())
    
    Q1 = np.percentile(data[:, i], 50)
    Q2 = np.percentile(data[:, i], 25)
    Q3 = np.percentile(data[:, i], 75)
    
    IQR = Q3-Q2
    innerFBound = IQR * 1.5
    innerFSmall = Q1 - innerFBound
    innerFLarge = Q3 + innerFBound
    
    #5 innerFenceBoundSmall
    arr.append(innerFSmall)
    
    #6 innerFenceBoundLarge
    arr.append(innerFLarge)   
    
    #7 min
    arr.append(data[:,i].min())
    
    #8 max
    arr.append(data[:,i].max())
    
    K = data[:, i]
   
    outliersLow = (K < innerFSmall) * K
    outliersHigh = (K > innerFLarge) * K

    arr.insert(11, outliersLow)
    arr.insert(12, outliersHigh)
    
    summary_Array.insert(j, arr)
   
    arr = []
    
    j += 1

del summary_Array[0]
del summary_Array[1]
del summary_Array[2]
del summary_Array[5]
del summary_Array[5]

#Sorting rows by death or not
temp1 = np.argsort(X[:, 12], axis=0)
SortedSurvival = X[temp1.ravel(), :]

#S = survived patients, D = dead patients
S, D = SortedSurvival[0:203, :], SortedSurvival[203:, :]

#Selecting saught after values
A, B = S[:, 7], D[:, 7]

values = A, B

#Box plot
fig, ax1 = plt.subplots()
ax1.boxplot(X[:, 8], vert=False)
ax1.set_title("Box-plot of " + attributeNames[8], fontsize=15)
ax1.set_xlabel("mEq/L", fontsize=15)
ax1.set_yticklabels([])
#plt.xticks(range(1,3), (0, 1))
#ax1.set_xticklabels(['Censored', 'Dead'], fontsize=15)

#t = np.sort(X[:, 0])
nbins = 30

fig, ax2 = plt.subplots()
ax2.hist(values, bins=nbins, histtype='bar')
ax2.set_title("Distribution of " + attributeNames[7] + " vs " + attributeNames[12], fontsize=15)
ax2.set_xlabel(attributeNames[7], fontsize=15)
ax2.set_ylabel("Patient count", fontsize=15)
ax2.legend(['Censored', 'Dead'])

plt.show()