# Import packages
import pandas as pd
import numpy as np
from scipy import stats
import itertools
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score

#print(adjusted_rand_score([32,119,8],[146,119,68]))

t = []
p = []
for x in range(114+8):
    t.append(x-(x-1))
for x in range(119):
    t.append(x-(x-2))
for x in range(32+60):
    t.append(x-(x-3))
print(t)
for x in range(146):
    p.append(x-(x-1))
for x in range(119):
    p.append(x-(x-2))
for x in range(68):
    p.append(x-(x-3))

print(p)
print(adjusted_rand_score(t,p))
# This document is created by Joel Madsen and Max Thrane, as a helper document for the exam.

# Functions for the exam:

# ONLY WORKS IF THERES 2 CLASSES!!!!
# calculate P(A|B) given P(A), P(B|A), P(B|not A) - Returns decimal value
def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):
    # calculate P(not A)
    not_a = 1 - p_a
    # calculate P(B)
    p_b = p_b_given_a * p_a + p_b_given_not_a * not_a
    # calculate P(A|B)
    p_a_given_b = (p_b_given_a * p_a) / p_b
    return p_a_given_b

#print(bayes_theorem(0.01, 0.97, 0.03))


# print(bayes_theorem(3/11,1/3,1/8))

# Calculates the variance explained by given principal components, arguments are two lists
# l1 = PCA components to check, l2 = all PCA components
def varianceExplainedByPCADirections(l1, l2):
    suml1 = 0
    for x in l1:
        suml1 += pow(x, 2)
    suml2 = 0
    for y in l2:
        suml2 += pow(y, 2)

    return suml1 / suml2 * 100

#print(varianceExplainedByPCADirections([30.19],[30.19,16.08,11.07,5.98]))
#print(varianceExplainedByPCADirections([30.19,16.08],[30.19,16.08,11.07,5.98]))
#print(varianceExplainedByPCADirections([30.19,16.08,11.07],[30.19,16.08,11.07,5.98]))
#print(varianceExplainedByPCADirections([5.98],[30.19,16.08,11.07,5.98]))

# Example
# print(varianceExplainedByPCADirections([4.69, 1.92], [22.44, 13.06, 9.24, 4.69, 1.92]))
# print(varianceExplainedByPCADirections([19.64,6.87,3.26,2.3],[19.64,6.87,3.26,2.3,1.12]))
# print(varianceExplainedByPCADirections([13.5],[13.5,7.6,6.5,5.8,3.5,2]))


# Calculates the average distance of a list l1, init = the point to calculate the distance from
def averageDistance(init, l1):
    l1_len = len(l1)
    sum_l1 = 0
    for x in l1:
        sum_l1 += np.sqrt(pow(np.abs(init - x), 2))

    return sum_l1 / l1_len


# Calculates the K nearest neighbor density. K = neighbors, distance = summed distances.
def knnDensity(distance, K):
    return 1 / ((1 / K) * (distance))

# Calculates the ARD for a list of distances for K neighbors
# the list of distances are given as:
# first entry is knn for the observation you're interested in
# next is knn for knn from the interested observation
def knnARD(distList, k):
    d1 = knnDensity(distList[0], k)
    sum = 0
    for i in range(1, len(distList)):

        sum += knnDensity(distList[i], k)

    return d1/((1/k) * sum)


#print(knnARD([75+125,75+51,125+51],2))

# Example
# print(knnARD([1.7 + 2.2, 1.8 + 1.7, 0.9 + 2.1], 2))
# print(knnARD([0.9 + 1, 1 + 1.3, 1.3 + 0.9], 2))
# print(knnARD([6.8+7.66,2.50+2.66,1.05+0.56],2))




# From https://gist.github.com/ramhiser/c990481c387058f3cce7
# Calculates the Jaccard similarity between two sets of clustering labels!
def jaccard(labels1, labels2):
    """
    Computes the Jaccard similarity between two sets of clustering labels.
    The value returned is between 0 and 1, inclusively. A value of 1 indicates
    perfect agreement between two clustering algorithms, whereas a value of 0
    indicates no agreement. For details on the Jaccard index, see:
    http://en.wikipedia.org/wiki/Jaccard_index
    Example:
    labels1 = [1, 2, 2, 3]
    labels2 = [3, 4, 4, 4]
    print jaccard(labels1, labels2)
    @param labels1 iterable of cluster labels
    @param labels2 iterable of cluster labels
    @return the Jaccard similarity value
    """
    n11 = n10 = n01 = 0
    n = len(labels1)
    # TODO: Throw exception if len(labels1) != len(labels2)
    for i, j in itertools.combinations(range(n), 2):
        comembership1 = labels1[i] == labels1[j]
        comembership2 = labels2[i] == labels2[j]
        if comembership1 and comembership2:
            n11 += 1
        elif comembership1 and not comembership2:
            n10 += 1
        elif not comembership1 and comembership2:
            n01 += 1
    return float(n11) / (n11 + n10 + n01)

# print(jaccard([1,1,2,1,1,1,1,2,2,0], [0,0,1,1,1,2,2,2,2,2]))
# print(jaccard([1,1,1,1,1,1,1,1,0,0,1],[1,1,1,1,1,1,1,1,0,0,0]))


# Calculates the confidence interval, df is the number of observations/tests
def confidenceInterval(alpha, df, zhat, std):
    return stats.t.interval(alpha=alpha, df=df - 1, loc=zhat, scale=std)
# Example:
# print(confidenceInterval(0.95, 5, 0.221, 0.054))

# Calculate the projection of a point onto weights
def multinomialWeightCalc(point, weights):
    y = np.array([1, point[0], point[1]])
    toprint = []
    j = 0
    p = 1
    for i in weights:
            npi = np.array(i)
            f = np.dot(y, npi)
            toprint.append(f)
            j += 1
            if j % 3 == 0:
                print(f"Option {p}: {toprint}")
                toprint = []
                p += 1

# Example
# multinomialWeightCalc([-0,-1],[[-0.77,-5.54,0.01],[0.26,-2.09,-0.03],[0,0,0],[0.51,1.65,0.01],[0.1,3.8,0.04],[0,0,0]])

