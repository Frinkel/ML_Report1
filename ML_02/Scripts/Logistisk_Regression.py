import pandas as pd
import numpy as np
import sklearn.linear_model as lm

from main import *

X = data[:, [0,2,4,6,7,8]]
y = data[:,12]

#attributeNames = np.asarray(df.columns[range(0,13)])

N, M = X.shape

model = lm.LogisticRegression()
model = model.fit(X,y)

y_est = model.predict(X)
y_est_dead_prob = model.predict_proba(X)[:, 1]
y_est_alive_prob = model.predict_proba(X)[:, 0]

x_class = model.predict_proba(X)[0,1]
misclass_rate = np.sum(y_est != y) / float(len(y_est))

print('\nProbability of given patient being dead: {0:.4f}'.format(x_class))
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))