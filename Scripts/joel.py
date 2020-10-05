# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import *
import enum
import scipy.stats as st


from matplotlib.collections import LineCollection


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

# Joel has features 0, 1, 2, 3, 4

# Plots 2 features with death event as color
def plot_with_deathevent(fdata, i, j):
    for c in range(fdata.shape[0]):
        class_mask = fdata[:, 12] == c
        plt.plot(fdata[class_mask, i], fdata[class_mask, j], 'o', alpha=.3)
    plt.legend(["Survivor", "Dead"])
    plt.xlabel(Feature(i).name)
    plt.ylabel(Feature(j).name)
    plt.show()


# Get summary statistics for features 0 to N
def extractDataInformation(N):
    for x in range(N):
        feat = Feature(x)
        print(f"## {feat.name.upper()} ##")
        print(f"    Mean: {np.mean(data[:, feat.value])}")
        print(f"    Median: {np.median(data[:, feat.value])}")
        print(f"    STD: {np.std(data[:, feat.value])}")
        print(f"    Variance {np.var([data[:, feat.value]])}")
        print(f"    Correlation with age: {np.corrcoef(data[:, feat.value], data[:, Feature.age.value])[1][0]}")
        print(f"    Covariance with age: {np.cov(data[:, feat.value], data[:, Feature.age.value])[1][0]}")
        print("\n")


# Plot a correlation matrix, with histograms in the diagonal
def correlationMatrix(data, labels):
    fig, axs = plt.subplots(data.shape[1], data.shape[1])
    fig.suptitle("Correlation Matrix Plot", size = 20)
    corMul = 3.5
    for x in range(data.shape[1]):
        for y in range(data.shape[1]):
            cor = np.corrcoef(data[:, x], data[:, y])[1][0]
            if (x == y):
                axs[x, y].hist(data[:,x], density=True, bins=10, rwidth=0.9)
            elif(x < y):
                axs[x, y].scatter(data[:, y], data[:, x], s=8, color = [1-np.absolute(corMul*cor), np.absolute(corMul*cor),0])
            else:
                axs[x, y].scatter(data[:, x], data[:, y], s=8, color = [1-np.absolute(corMul*cor), np.absolute(corMul*cor),0])

            if(x < 1):
                axs[x, y].set_title(f"{Feature(labels[y]).name}")

            if(y < 1):
                axs[x, y].set_title(f"{Feature(labels[x]).name}", x = -0.9, y = 0.3, loc = "left")

            axs[x, y].xaxis.set_visible(False)
            axs[x, y].yaxis.set_visible(False)
            axs[x, y].label_outer()
    plt.show()


# Create histogram plot
def plotHistograms(data, labels):

    fig, axs = plt.subplots(2, int(data.shape[1]/2))
    fig.suptitle("Histogram of chosen features", size=20)
    N = 0
    for i in range(2):
        for j in range(int(data.shape[1]/2)):
            axs[i][j].hist(data[:, N], density=True, bins=10, rwidth=0.9, label = Feature(labels[N]).name)

            mn, mx = axs[i][j].get_xlim()
            axs[i][j].set_xlim(mn, mx)
            lspace = np.linspace(mn, mx, 301)
            gausianFit = st.gaussian_kde(data[:,N])
            axs[i][j].plot(lspace, gausianFit.pdf(lspace), label="Gaussian fit", color = "red")
            axs[i][j].legend(loc="upper right")
            axs[i][0].set_ylabel('Probability', size = 12)
            axs[i][j].set_xlabel(Feature(labels[N]).name, size = 12)
            #axs[i].set_title("Histogram");

            N += 1


    plt.show()

# Big scatter plot
def scatterStandardizedALL(data, labels):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    for i in range(data_scaled.shape[1]):
        plt.plot(data_scaled[:, i], 'o', alpha=.3)
    plt.legend(labels)
    #plt.xlabel(Feature(i).name)
    #plt.ylabel(Feature(j).name)
    plt.show()




# -------- RUN

# Get basic summery statistics for every feature
extractDataInformation(13)


# Create an array holding the values of the continuous data columns
vals = [0,2,4,6,7,8]
# Create new np array containing only the continuous data
continuousData = data[:,vals]

# Check if the features are normally distributed
for i in range(continuousData.shape[1]):
    print(st.anderson(continuousData[:,i], dist = "norm"))

# Plot the Correlation matrix
correlationMatrix(continuousData, vals)

# Plot the Histograms
plotHistograms(continuousData, vals)

# Get the correlation matrix using pandas
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(df.corr())
