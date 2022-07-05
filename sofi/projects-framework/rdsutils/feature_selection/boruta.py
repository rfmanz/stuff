"""
Simplified implementation of the Bortua feature selection
algorithm.
"""

import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_random_state

class Boruta(BaseEstimator, TransformerMixin):
    """
    Boruta feature selection algorithm.
    """
    def __init__(self, estimator, max_iter=20, min_features=1, thresh=0.25, 
                 drop_at=None, random_state=None, verbose=0):
        self.estimator = estimator
        self.max_iter = max_iter
        self.min_features = min_features
        self.thresh = thresh
        self.drop_at = drop_at
        self.random_state = random_state
        self.verbose = verbose
        
        if self.drop_at is None:
            self.drop_at = min(int(-1 * round(self.max_iter / 5)), -1)
            
        self.random_state = check_random_state(self.random_state)

    def fit(self, X, y, **kwargs):
        """
        Fits the Boruta feature selection with the provided estimator.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        """

        return self._fit(X, y, **kwargs)

    def transform(self, X):
        """
        Reduces the input X to the features selected; by Boruta.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features which were
            selected by Boruta.
        """

        return self._transform(X)

    def fit_transform(self, X, y, **kwargs):
        """
        Fits Boruta, then reduces the input X to the selected features.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        weak: boolean, default = False
            If set to true, the tentative features are also used to reduce X.
        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features which were
            selected by Boruta.
        """

        self._fit(X, y, **kwargs)
        return self._transform(X)

    def _fit(self, X, y, **kwargs):
        """    
        """

        self.scores = np.zeros(X.shape[-1])
        self.imps = np.zeros((self.max_iter, X.shape[-1]))
        
        self.features = np.array(range(X.shape[-1]))
        self.remaining_features = self.features.copy()

        _iter = 1
        while _iter <= self.max_iter and len(self.remaining_features) > self.min_features:
            if self.verbose:
                print('Running iteration number {}.'.format(str(_iter)), end="\r")
                
            X_shadow = self._get_x_with_shadow_attributes(X)
            
            model = self.estimator.fit(X_shadow, y, **kwargs)
            
            f_imps = np.array(model.feature_importances_[:len(self.remaining_features)])
            self.imps[_iter-1, self.remaining_features] = f_imps
            
            max_shadow_imp = max(model.feature_importances_[len(self.remaining_features):])
            
            lb = (1 - self.thresh) * max_shadow_imp
            score_it = np.where(f_imps < lb, -1, f_imps)
            
            ub = (1 + self.thresh) * max_shadow_imp
            score_it = np.where(score_it > ub, 1, score_it)
            score_it = np.where((score_it <= ub) & (score_it >= lb), 0, score_it)
            
            self.scores[self.remaining_features] += score_it
            self.remaining_features = self.features[self.scores > self.drop_at]
            
            if self.verbose:
                print('Completed iteration number {}.'.format(str(_iter)), end="\r")
            _iter += 1
            
    
    def _get_x_with_shadow_attributes(self, X):
        """
        Return a data with shuffled data.
        """
        
        X_shadow = X[:, self.remaining_features]
        X_shadow_copy = X_shadow.copy()
        self.random_state.shuffle(X_shadow_copy)
        return np.concatenate([X_shadow, X_shadow_copy], axis=1)
    
    def _transform(self, X):
        """
        """
        return X[:, self.remaining_features]
