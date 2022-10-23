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
    X, y = make_dataset2(n_points, random_state)
    
    # sample 1200 points to use as training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAINING, test_size=TESTING, random_state = random_state)
    test_depth = (1, 2, 4, 8, None)
    
    # Decision boundary
    record = []
    for depth in test_depth:
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train, y_train)
        record.append((clf.get_depth(), clf.score(X_train, y_train), clf.score(X_test, y_test)))
        # plot the score for each depth using record
        plot_boundary("boundary" + str(clf.get_depth()), clf, X_train, y_train, title="Boundary of decision tree classifier " + str(clf.get_depth()))
    
    # create a side by side bar plot with the score for each depth
    plt.bar(np.arange(len(record)), [x[1] for x in record], width=0.2, label="Training score")
    plt.bar(np.arange(len(record)) + 0.2, [x[2] for x in record], width=0.2, label="Testing score")
    plt.xticks(np.arange(len(record)) + 0.1, [x[0] for x in record])
    plt.xlabel("Depth")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig("scoreDT.pdf")
    
    # draw the decision tree
    from sklearn.tree import export_graphviz
    from six import StringIO  
    from IPython.display import Image  
    import pydotplus
    clf = DecisionTreeClassifier(max_depth=4) 
    clf.fit(X_train, y_train)
    record.append((clf.get_depth(), clf.score(X_train, y_train), clf.score(X_test, y_test)))
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('treeDT.png')
    Image(graph.create_png())

    # Q 1.2
    final_res = []
    for depth in test_depth:
        record = []
        for _ in range(20):
            X, y = make_dataset2(n_points)
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAINING, test_size=TESTING, random_state = random_state)    
            clf = DecisionTreeClassifier(max_depth=depth) 
            clf.fit(X_train, y_train)
            record.append((clf.get_depth(), clf.score(X_train, y_train), clf.score(X_test, y_test)))
        final_res.append((clf.get_depth(), np.mean([x[2] for x in record]), np.std([x[2] for x in record])))
    print(final_res)
    
    # Q 4.1 and 4.2
    final_res = []
    ds = 2
    for depth in test_depth:
        record = []
        for _ in range(100):
            if (ds == 1):
                X, y = make_dataset1(n_points)
            else:
                X, y = make_dataset2(n_points)
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAINING, test_size=TESTING, random_state = random_state)    
            clf = DecisionTreeClassifier(max_depth=depth) 
            clf.fit(X_train, y_train)
            record.append((clf.get_depth(), clf.score(X_train, y_train), clf.score(X_test, y_test)))
        final_res.append((clf.get_depth(), np.mean([x[2] for x in record]), np.std([x[2] for x in record])))
    # choose the best value for the testing set in the final_res
    best_depth = max(final_res, key=lambda x: x[1])
    print("Best depth for dataset" + str(ds) + " is: ", best_depth)
    
if __name__ == "__main__":
    main()
