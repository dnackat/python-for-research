#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 15:51:18 2018

@author: dileepn
"""
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

n = 100
beta_0 = 5
beta_1 = 2
np.random.seed(1)
# 'n' random variables that are distributed between 0 and 1
x = 10 * ss.uniform.rvs(size=n)
# Add random noise (gaussian) with ss.norm (mean=0, n=100)
y = beta_0 + beta_1 * x + ss.norm.rvs(loc=0, scale=1, size=n)

plt.figure()
plt.plot(x, y, 'o', ms=5) # ms = marker size
xx = np.array([0,10])
plt.plot(xx, beta_0 + beta_1 * xx)
plt.xlabel('$x$')
plt.ylabel('$y$')


# Simple linear regression
def compute_rss(y_estimate, y): 
  return sum(np.power(y-y_estimate, 2))
 
def estimate_y(x, b_0, b_1): 
  return b_0 + b_1 * x 

rss = compute_rss(estimate_y(x, beta_0, beta_1), y)

# Least squares estimation in code
rss = []
slopes = np.arange(-10,15,0.01)
for slope in slopes:
    rss.append(np.sum((y - beta_0 - slope * x)**2))

ind_min = np.argmin(rss) # index where rss has its min. value
print("Estimate for the slope: ", slopes[ind_min])

# Plot figure
plt.figure()
plt.plot(slopes, rss)
plt.xlabel("Slope")
plt.ylabel("RSS")

# Simple linear regression in code (no clue what this part means!)
import statsmodels.api as sm
X = sm.add_constant(x) # Add the intercept term
mod = sm.OLS(y, X)
est = mod.fit()
print(est.summary())

# Scikit-learn for linear regression
n = 500
beta_0 = 5
beta_1 = 2
beat_2 = -1
np.random.seed(1)
x_1 = 10 * ss.uniform.rvs(size=n)
x_2 = 10 * ss.uniform.rvs(size=n)
y = beta_0 + beta_1 * x_1 + beat_2 * x_2 + ss.norm.rvs(loc=0, scale=1, size=n)

X = np.stack([x_1, x_2], axis=1) # To construct input matrix, X from x vectors

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y, c=y)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$y$")

from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=True)
lm.fit(X, y)

# lm.intercept_, lm.coef_[0], lm.coef_[1] for b0, b1, and b2
# lm.predict(np.array([2,4]).reshape(1,-1)) -  reshape to avoid size mismatch error
# (1, -1) is used for single sample (x1, x2)
# lm.score(X,y) --> prediction accuracy

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)

lm = LinearRegression(fit_intercept=True)

lm.fit(X_train, y_train)
print(lm.score(X_test, y_test))