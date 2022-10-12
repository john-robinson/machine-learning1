"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from data import make_dataset1, make_dataset2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from plot import plot_boundary

# (Question 2)
# Constants
TRAINING= 1200
TESTING = 300

def main():
    n_points = 1500
    random_state = 5
    X, y = make_dataset2(n_points, random_state) # x = points, y = classe
        
    # sample 1200 points to use as training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAINING, test_size=TESTING, random_state = random_state)
    n_neighbors = (1, 5, 25, 125, 625, 1200)
    
    # train the model
    record = []
    for n in n_neighbors:
        model = KNeighborsClassifier(n_neighbors=n) # Tuning hyperparameters
        model.fit(X_train, y_train)     # Training model
        record.append((n, model.score(X_train, y_train), model.score(X_test, y_test))) # Testing model
        # plot the score for each depth using record
        plot_boundary("boundaryKNN" + str(n), model, X_train, y_train, title="Boundary of k-Neighbors classifier " + str(n))

if __name__ == "__main__":
    main()
