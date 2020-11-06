# Import packages
import pandas as pd
import numpy as np
import torch as t
from scipy import stats

# Load Cardiovascular disease data using pandas
filename = '../Data/CVD-data.csv'
df = pd.read_csv(filename)

# Convert data from dataframe (df) to a numpy array
data = df.to_numpy()

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

# Standardize the data, z-score
# standData = (data - np.mean(data)) / np.std(data)
standData = stats.zscore(data)

