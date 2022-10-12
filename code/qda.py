"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.model_selection import train_test_split



class QuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):


    def fit(self, X, y, lda=False):
        """Fit a linear discriminant analysis model using the training set
        (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        self.lda = lda

        
        # ====================
        # mean vector of X for each class
        self.means = np.array([np.mean(X[y == b], axis=1) for b in np.unique(y)])
        
        # covariance matrix of X for each class
        if self.lda:
            self.covs = np.array([np.cov(X[y == b], rowvar=False) for b in np.unique(y)])
        else: # QDA
            self.covs = np.array([np.cov(X[y == b], rowvar=False) for b in np.unique(y)])
            
        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values. 
        """
        # Vecteur de classe
        # ====================
        # TODO your code here.
        # ====================

        # argmax(k, self.predict_proba(X))

        pass

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        # ====================
        # TODO your code here.
        # ====================

        pass

if __name__ == "__main__":
    from data import make_dataset1
    from plot import plot_boundary
    TRAINING= 1200
    TESTING = 300
    n_points = 1500
    random_state = 5
    X, y = make_dataset1(n_points, random_state=random_state)
    
    # sample 1200 points to use as training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAINING, test_size=TESTING, random_state = random_state)
    
    ncl = QuadraticDiscriminantAnalysis()
    ncl.fit(X_train, y_train)
    print(ncl.means[0])