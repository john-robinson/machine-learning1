"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

from pickle import NONE
import numpy as np
from matplotlib import pyplot as plt

from data import make_dataset1, make_dataset2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from plot import plot_boundary


# (Question 1)

# Constants
TRAINING= 1200
TESTING = 300

def main():
    n_points = 1500
    random_state = 5
    X, y = make_dataset2(n_points, random_state) # x = points, y = classe
    
    # sample 1200 points to use as training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAINING, test_size=TESTING, random_state = random_state)
    test_depth = (1, 2, 4, 8, None)
    
    # train the model
    record = []
    for depth in test_depth:
        clf = DecisionTreeClassifier(max_depth=depth) # Tuning hyperparameters
        clf.fit(X_train, y_train)     # Training model
        record.append((clf.get_depth(), clf.score(X_train, y_train), clf.score(X_test, y_test))) # Testing model
        # plot the score for each depth using record
        plot_boundary("boundary" + str(clf.get_depth()), clf, X_test, y_test, title="Boundary of decision tree classifier " + str(clf.get_depth()))

if __name__ == "__main__":
    main()
