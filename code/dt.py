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
        plot_boundary("boundary" + str(clf.get_depth()), clf, X_train, y_train, title="Boundary of decision tree classifier " + str(clf.get_depth()))
    
    # create a side by side bar plot with the score for each depth
    # plt.bar(np.arange(len(record)), [x[1] for x in record], width=0.2, label="Training score")
    # plt.bar(np.arange(len(record)) + 0.2, [x[2] for x in record], width=0.2, label="Testing score")
    # plt.xticks(np.arange(len(record)) + 0.1, [x[0] for x in record])
    # plt.xlabel("Depth")
    # plt.ylabel("Score")
    # plt.legend()
    # plt.savefig("score.pdf")

    final_res = []
    for depth in test_depth:
        record = []
        for _ in range(20):
            X, y = make_dataset2(n_points)
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAINING, test_size=TESTING, random_state = random_state)    
            clf = DecisionTreeClassifier(max_depth=depth) # Tuning hyperparameters
            clf.fit(X_train, y_train)     # Training model
            record.append((clf.get_depth(), clf.score(X_train, y_train), clf.score(X_test, y_test))) # Testing model
        final_res.append((clf.get_depth(), np.mean([x[2] for x in record]), np.std([x[2] for x in record])))
    print(final_res)
if __name__ == "__main__":
    main()
