# Import packages
import pandas as pd
import numpy as np
from scipy import stats
import itertools


# This document is created by Joel Madsen, Lasse Damgård and Max Thrane, as a helper document for the exam.

# Functions for the exam:

# calculate P(A|B) given P(A), P(B|A), P(B|not A) - Returns decimal value
def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):
    # calculate P(not A)
    not_a = 1 - p_a
    # calculate P(B)
    p_b = p_b_given_a * p_a + p_b_given_not_a * not_a
    # calculate P(A|B)
    p_a_given_b = (p_b_given_a * p_a) / p_b
    return p_a_given_b


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


# print(varianceExplainedByPCADirections([4.69, 1.92], [22.44, 13.06, 9.24, 4.69, 1.92]))

# Calculates the average distance of a list l1, init = the point to calculate the distance from
def averageDistance(init, l1):
    l1_len = len(l1)
    sum_l1 = 0
    for x in l1:
        sum_l1 += np.sqrt(pow(np.abs(init - x), 2))

    return sum_l1 / l1_len


# Calculates the K nearest neighbor density. K = neighbors, distance = the average distance.
def knnDensity(distance, K):
    return 1 / (1 / K * distance)


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



# Calculates the confidence interval, df is the number of observations/tests
def confidenceInterval(alpha, df, zhat, std):
    return stats.t.interval(alpha=alpha, df=df - 1, loc=zhat, scale=std)
# Example:
# print(confidenceInterval(0.95, 5, 0.221, 0.054))