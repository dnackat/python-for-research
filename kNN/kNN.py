#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 23:41:09 2018

@author: dileepn
"""

import numpy as np
import random

def distance(point1, point2):
    """ Finds the Euclidean distance between points p1 and p2. """

    distance = np.sqrt(np.power((point2[0] - point1[0]), 2.0) + \
                np.power((point2[1] - point1[1]), 2.0))
    return distance

def majority_vote(votes):
    """ Returns the winner based on majority vote. """
    
    vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1
            
    winners = []
    max_counts = max(vote_counts.values())
    for vote, count in vote_counts.items():
        if count == max_counts:
            winners.append(vote)
            
    return random.choice(winners)  # In case there are multiple winners

import scipy.stats as ss
import matplotlib.pyplot as plt

def majority_vote_short(votes):
    """ Returns the winner based on majority vote using mode in Scipy. """
    
    mode, count = ss.mstats.mode(votes)
    return mode

def find_nearest_neighbors(p, points, k = 5):
    """ Find the k nearest neighbors of point 'p' """
    
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances) # To sort distance from low to high
    return ind[:k] # Pick the k nearest neighbors
    
def knn_predict(p, points, outcomes, k = 5):
    """ Predict the class of p based on k nearest neighbors"""
    
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote(outcomes[ind])

def generate_synth_data(n = 50):
    """ Create two sets of points from bivariate normal distribution. """
    points = np.concatenate((ss.norm(0, 1).rvs((n, 2)), ss.norm(1, 1).rvs((n, 2))), axis = 0)
    outcomes = np.concatenate((np.repeat(0, n), np.repeat(1, n)))
    return (points, outcomes)

def make_prediction_grid(predictors, outcomes, limits, h, k):
    """ Classify each point on the prediction grid. """
    
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)
    
    prediction_grid = np.zeros(xx.shape, dtype = int)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            prediction_grid[j, i] = knn_predict(p, predictors, outcomes, k)
            # j, i because we want to assign y values to rows of the array
            
    return (xx, yy, prediction_grid)

def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
predictors = iris.data[:, 0:2]
outcomes = iris.target

plt.plot(predictors[outcomes==0][:,0], predictors[outcomes==0][:,1], "ro")
plt.plot(predictors[outcomes==1][:,0], predictors[outcomes==1][:,1], "go")
plt.plot(predictors[outcomes==2][:,0], predictors[outcomes==2][:,1], "bo")
plt.savefig("iris.pdf")

k=5; filename="iris_grid.pdf"; limits=(4,8,1.5,4.5); h = 0.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(predictors, outcomes)
sk_predictions = knn.predict(predictors)
my_predictions = np.array([knn_predict(p, predictors, outcomes, 5) for p \
                           in predictors])

print(100 * np.mean(sk_predictions == my_predictions))
print("{:.1f}".format(100 * np.mean(sk_predictions == outcomes)))
print("{:.1f}".format(100 * np.mean(my_predictions == outcomes)))

