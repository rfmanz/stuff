"""
Base GRMlab model.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import numpy as np

from abc import ABCMeta, abstractmethod

from sklearn.utils import check_X_y
from sklearn.utils.metaestimators import if_delegate_has_method

from ..core.base import GRMlabBase
from ..core.exceptions import NotFittedException


def _check_feature_selection_estimator(feature_selection, estimator, verbose):
    """
    Check the availability of the feature selection and estimator objects.
    If only one of them is available, print info message (if verbose).
    """
    if (feature_selection is None) and (estimator is None):
        raise ValueError("At least an feature selection or an estimator is "
                         "required")

    if feature_selection is None and verbose:
        print("The feature selection is not available.")
    elif estimator is None and verbose:
        print("The estimator is not available.")


class GRMlabModel(GRMlabBase, metaclass=ABCMeta):
    """
    Base class for all models in GRMlab.

    Parameters
    ----------
    name : str
        Model name.

    description: str or dict (default=None)
        Information about the model, for example specifiy its purpose and the
        feature selection and estimator methods and parameters.

    feature_names : array, shape (n_features, ) or None (default=None)
        The name of the features.

    feature_selection : object (default=None)
        A feature selection algorithm with methods ``fit``, ``fit_transform``
        and ``transform``, and attribute ``support_``.

    estimator : object (default=None)
        A supervised learning estimator with a ``fit`` method.

    verbose : int or boolean (default=False)
        Controls verbosity of output.

    Notes
    -----
    This is an abstract class not directly callable.
    """
    def __init__(self, name, description=None, feature_names=None,
                 feature_selection=None, estimator=None, verbose=False):
        self.name = name
        self.description = description
        self.feature_names = feature_names
        self.feature_selection = feature_selection
        self.estimator = estimator
        self.verbose = verbose

        # auxiliary data
        self._n_samples = None
        self._n_features = None

        # flags
        self._is_feature_selection_fitted = False
        self._is_estimator_fitted = False

    def fit(self, X, y, sample_weight=None):
        """
        Fit the feature selection method and pass transformed data to fit
        the estimator. If no feature selection is provided, the estimator is
        fitted with the original training samples without prior transformation.
        If no estimator is provided, only feature selection is performed.
        Otherwise, if not feature selection nor estimator is provided, an
        exception is raised.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            The training input samples.

        y : array-like, shape = (n_samples)
            The target values.

        sample_weight : array-like of shape = (n_samples) default=None
            Array of weights that are assigned to individual samples. If not
            provided, then each sample is given unit weight. Be aware that
            not all estimators accept weights on fit.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, "csc")

        if sample_weight is not None:
            X, sample_weight = check_X_y(X, sample_weight, "csc")
            kwargs = {"sample_weight": sample_weight}
        else:
            kwargs = {}

        self._n_samples, self._n_features = X.shape

        _check_feature_selection_estimator(self.feature_selection,
                                           self.estimator, self.verbose)

        if self.feature_selection is None:
            self.estimator.fit(X, y, **kwargs)
            self._is_estimator_fitted = True
        elif self.estimator is None:
            self.feature_selection.fit(X, y)
            self._is_feature_selection_fitted = True
        else:
            self.estimator.fit(self.feature_selection.fit_transform(X, y), y,
                               **kwargs)
            self._is_feature_selection_fitted = True
            self._is_estimator_fitted = True

        return self

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        """
        Reduce X to the selected features, if feature selection is provided,
        and then predict using the underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        if not self._is_estimator_fitted:
            raise NotFittedException(self)

        if not self._is_feature_selection_fitted:
            return self.estimator.predict(X)

        return self.estimator.predict(self.feature_selection.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        """
        Reduce X to the selected features, if feature selection is provided,
        and then return the score of the underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        if not self._is_estimator_fitted:
            raise NotFittedException(self)

        if not self._is_feature_selection_fitted:
            self.estimator.score(X, y)

        return self.estimator.score(self.feature_selection.transform(X), y)

    @if_delegate_has_method(delegate='estimator')
    def decision_function(self, X):
        """
        Compute the decision function of X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : array, shape = [n_samples, n_classes] or [n_samples]
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification produce an array of shape
            [n_samples].
        """
        if not self._is_estimator_fitted:
            raise NotFittedException(self)

        if not self._is_feature_selection_fitted:
            self.estimator.decision_function(X)

        return self.estimator.decision_function(
            self.feature_selection.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        if not self._is_estimator_fitted:
            raise NotFittedException(self)

        if not self._is_feature_selection_fitted:
            return self.estimator.predict_proba(X)

        return self.estimator.predict_proba(
            self.feature_selection.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_log_proba(self, X):
        """
        Predict class log-probabilities for X.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        if not self._is_estimator_fitted:
            raise NotFittedException(self)

        if not self._is_feature_selection_fitted:
            return self.estimator.predict_log_proba(self, X)

        return self.estimator.predict_log_proba(
            self.feature_selection.transform(X))

    def get_support(self, output_names=False, indices=False):
        """
        Get a mask, or the names of the features selected.

        Parameters
        ----------
        output_names : boolean (default=False)
            If True and ``feature_names`` is not None, the return value will be
            an array of strings, rather than a boolean mask.

        indices : boolean (default False)
            If True, the return value will be an array of integers, rather
            than a boolean mask.

        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If output_names is True, this is a string array of shape [# output
            features]. If output_names is False, this is a boolean array of
            shape [# input features], in which an element is True iff its
            corresponding feature is selected for retention.
        """
        if self._is_feature_selection_fitted:
            if not hasattr(self.feature_selection, "support_"):
                raise RuntimeError('The Feature selection method does not '
                                   'expose "support_" attribute')
            if output_names:
                if self.feature_names is not None:
                    array_features = np.asarray(self.feature_names)
                    return array_features[self.feature_selection.support_]
                elif self.feature_selection.feature_names is not None:
                    array_features = np.asarray(
                        self.feature_selection.feature_names)
                    return array_features[self.feature_selection.support_]
                else:
                    return np.where(
                        self.feature_selection.support_)[0].astype(str)
            elif indices:
                return np.where(self.feature_selection.support_)[0]
            else:
                return np.asarray(self.feature_selection.support_)
        else:
            if output_names:
                if self.feature_names is not None:
                    return np.asarray(self.feature_names)
                else:
                    return np.where(np.ones(
                        self._n_features).astype(np.bool))[0].astype(str)
            elif indices:
                return np.where(np.ones(self._n_features).astype(np.bool))[0]
            else:
                return np.ones(self._n_features).astype(np.bool)

    def get_params_feature_selection(self, deep=True):
        """
        Get parameters for the feature selection.

        Parameters
        ----------
        deep : boolean (default=True)
            If True, will return the parameters for this model feature
            selection and contained subojects that are feature selection
            methods.

        Returns
        -------
        params : mapping of strings to any
            Parameter names mapped to their values.
        """
        if not hasattr(self.feature_selection, "get_params"):
            raise RuntimeError('The feature selection does not expose '
                               '"get_params" method')

        return self.feature_selection.get_params(deep)

    def get_params_estimator(self, deep=True):
        """
        Get parameters for the estimator.

        Parameters
        ----------
        deep : boolean (default=True)
            If True, will return the parameters for this model estimator
            and contained subojects that are estimators.

        Returns
        -------
        params : mapping of strings to any
            Parameter names mapped to their values.
        """
        if not hasattr(self.estimator, "get_params"):
            raise RuntimeError('The estimator does not expose "get_params" '
                               'method')

        return self.estimator.get_params(deep)

    def set_params_feature_selection(self, **params):
        """
        Set the parameters of the feature selection.

        Returns
        -------
        self
        """
        if not hasattr(self.feature_selection, "set_params"):
            raise RuntimeError('The feature selection does not expose '
                               '"set_params" method')

        self.feature_selection.set_params(**params)
        self._is_feature_selection_fitted = False

    def set_params_estimator(self, **params):
        """
        Set the parameters of the estimator.

        Returns
        -------
        self
        """
        if not hasattr(self.estimator, "set_params"):
            raise RuntimeError('The estimator does not expose "set_params" '
                               'method')

        self.estimator.set_params(**params)
        self._is_estimator_fitted = False

    @abstractmethod
    def metrics(self):
        """
        Abstract method to return model performance metrics.
        """
        raise NotImplementedError
