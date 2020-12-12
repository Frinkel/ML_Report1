# Import packages
import pandas as pd
import numpy as np
from scipy import stats


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

    return sum_l1/l1_len



# Calculates the K nearest neighbor density. K = neighbors, distance = the average distance.
def knnDensity(distance, K):
    return 1/(1/K*distance)