# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import *
import enum
import sys



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
class Feature(enum.Enum):
    age = 0
    anaemia = 1
    CPK = 2
    diabetes = 3
    eject_fraction = 4
    HBP = 5
    platelets = 6
    ser_creatine = 7
    ser_sodium = 8
    sex = 9
    smoking = 10
    time = 11
    death = 12


# Gets the age of observation 0
# print(data[:, 2])

# plt.plot(data[:, 0], data[:, 2], 'o')
# plt.title('Creatine explained by Age');
# plt.xlabel('Age');
# plt.ylabel('Creatine');
# plt.legend(['Individual'])
# plt.show()

# Joel has features 1, 2, 3, 4

def plot_with_deathevent(fdata, i, j):
    for c in range(fdata.shape[0]):
        class_mask = fdata[:, 12] == c
        plt.plot(fdata[class_mask, i], fdata[class_mask, j], 'o', alpha=.3)
    plt.legend(["Survivor", "Dead"])
    plt.xlabel(Feature(i).name)
    plt.ylabel(Feature(j).name)
    plt.show()

# plot_with_deathevent(data, 4, 8)

# Get summary statistics for features 1, 2, 3 and 4
def extractDataInformation(N):
    for x in range(N):
        feat = Feature(x + 1)
        print(f"## {feat.name.upper()} ##")
        print(f"    Mean: {np.mean(data[:, feat.value])}")
        print(f"    Median: {np.median(data[:, feat.value])}")
        print(f"    STD: {np.std(data[:, feat.value])}")
        print(f"    Variance {np.var([data[:, feat.value]])}")
        print(f"    Correlation with age: {np.corrcoef(data[:, feat.value], data[:, Feature.age.value])[1][0]}")
        print(f"    Covariance with age: {np.cov(data[:, feat.value], data[:, Feature.age.value])[1][0]}")
        print("\n")

    #plot_with_deathevent(data, 0, feat.value)


# Line skip
#print("\n")
#np.set_printoptions(threshold=sys.maxsize)
print(df.corr().to_numpy())


