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
    X, y = make_dataset2(n_points, random_state)
        
    # sample 1200 points to use as training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAINING, test_size=TESTING, random_state = random_state)
    n_neighbors = (1, 5, 25, 125, 625, 1200)
    
    # train the model
    record = []
    for n in n_neighbors:
        model = KNeighborsClassifier(n_neighbors=n) 
        model.fit(X_train, y_train)
        record.append((n, model.score(X_train, y_train), model.score(X_test, y_test)))
        # plot the score for each depth using record
        plot_boundary("boundaryKNN" + str(n), model, X_train, y_train, title="Boundary of k-Neighbors classifier " + str(n))

    #create a side by side bar plot with the score for each depth
    # plt.bar(np.arange(len(record)), [x[1] for x in record], width=0.2, label="Training score")
    # plt.bar(np.arange(len(record)) + 0.2, [x[2] for x in record], width=0.2, label="Testing score")
    # plt.xticks(np.arange(len(record)) + 0.1, [x[0] for x in record])
    # plt.xlabel("n_neighbors")
    # plt.ylabel("Score")
    # plt.legend()
    # plt.savefig("scoreKNN.pdf")
    
    final_res = []
    for n in n_neighbors:
        record = []
        for _ in range(5):
            X, y = make_dataset2(n_points)
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAINING, test_size=TESTING, random_state = random_state)    
            model = KNeighborsClassifier(n_neighbors=n)
            model.fit(X_train, y_train)
            record.append((n, model.score(X_train, y_train), model.score(X_test, y_test)))
        final_res.append((n, np.mean([x[2] for x in record]), np.std([x[2] for x in record])))
    print(final_res)

if __name__ == "__main__":
    main()
