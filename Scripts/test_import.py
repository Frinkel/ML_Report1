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


# Gets the age of observation 0
# print(data[:, 0])

# plt.plot(data[:, 0], data[:, 6], 'o')
# plt.show()

# Print certain type of data that is above a threshold
feature_type = 0
threshold = 600000


# Print certain type of data that is above a threshold
def threshold_extraction(fdata, ftype, fthreshold):
    for c in fdata:
        if (c[ftype] >= fthreshold):
            print(f"Age: {c[feature.age]}, Creatine: {c[2]}, Is a smoker: {bool(c[10])}, Woman/Man: {c[9]}")


# Cal culate the mean of a column
def mean(fdata, fcol):
    fmean = sum(fdata[:, fcol]) / fdata.shape[0]
    return fmean


print(f"Mean:  {mean(data, feature_type)}")
# threshold_extraction(data, feature_type, mean(data, feature_type))
