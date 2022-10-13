"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

from pickle import TRUE
from tkinter import FALSE
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal


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
        self.classes = np.unique(y)

        
        # ====================
        # mean vector of X for each class
        self.means = np.array([np.mean(X[y == b], axis=0) for b in np.unique(y)])
        
        # covariance matrix of X for each class
        if self.lda:
            self.covs = np.array([np.cov(X, rowvar=False) for _ in np.unique(y)])
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
        # argmax(k, self.predict_proba(X))
        return np.argmax(self.predict_proba(X), axis=1)

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

        # fk(x)
        var = np.array([multivariate_normal(mean=self.means[k], cov=self.covs[k]) for k in range(len(self.classes))])
            
            
        # initial prior is uniform
        prior = np.full(shape=len(self.classes), fill_value=1/len(self.classes))
        
        proba_per_class = np.empty((X.shape[0], len(self.classes)))
        for x in range(len(X)):
            for k in range(len(self.classes)):
                # calculer P(y = k|x)
                proba_per_class[x][k] = prior[k] * var[k].pdf(X[x]) / np.sum([prior[i] * var[i].pdf(X[x]) for i in range(len(self.classes))])
                
        return proba_per_class


if __name__ == "__main__":
    from data import make_dataset1
    from data import make_dataset2
    from plot import plot_boundary
    TRAINING= 1200
    TESTING = 300
    n_points = 1500
    random_state = 5
    LDA = False
    X, y = make_dataset1(n_points, random_state=random_state)
    
    # sample 1200 points to use as training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAINING, test_size=TESTING, random_state = random_state)
    
    ncl = QuadraticDiscriminantAnalysis()
    ncl.fit(X_train, y_train, lda = LDA)
    plot_boundary("boundary" + "LDA" if LDA else "boundary" + "QDA" , ncl, X_test, y_test)