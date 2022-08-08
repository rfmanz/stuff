"""
Scoring functions
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score

from ...core.base import GRMlabBase
from ..base import GRMlabModel
from ..metrics.classification import auc_pr
from ..metrics.classification import auc_roc
from ..metrics.classification import balanced_accuracy
from ..metrics.classification import gini
from ..metrics.classification import log_loss
from ..metrics.classification import youden_index
from ..metrics.regression import mean_absolute_error
from ..metrics.regression import mean_absolute_percentage_error
from ..metrics.regression import mean_percentage_error
from ..metrics.regression import mean_squared_error
from ..metrics.regression import mean_squared_percentage_error
from ..metrics.regression import median_absolute_error
from ..metrics.regression import r2_score


class Scorer(GRMlabBase, metaclass=ABCMeta):
    """
    Scorer class

    Parameters
    ----------
    metrics : array-like, shape = [n_metrics]
        The name of the metrics to include in the scoring function.

    weights : array-like, shape = [n_metrics] (default=None)
        The weights associated with metrics. If not given, all metrics are
        supposed to have equal weight. If weights do not sum up 1, an exception
        is raised.

    See also
    --------
    grmlab.modelling.base.GRMlabModel

    Notes
    -----
    This is an abstract class not directly callable.
    """
    def __init__(self, metrics, weights=None):
        self.metrics = metrics
        self.weights = weights

        self._is_pred_required = False
        self._is_pred_proba_required = False

        # Check weights
        if weights is not None:
            if len(weights) != len(metrics):
                raise ValueError

            if sum(weights) != 1.0:
                raise ValueError("sum of weights must equate 1.")

    @abstractmethod
    def _check_metrics(self):
        """Abstract method to Check if all metrics are valid."""
        raise NotImplementedError

    @abstractmethod
    def score(self):
        """Abstract method to compute scoring function."""
        raise NotImplementedError


METRICS_BINARY_CLASSIFICATION = {
    'auc_roc': {'proba': 1, 'greater_is_better': 1, "min": 0, "max": 1},
    'auc_pr': {'proba': 1, 'greater_is_better': 1, "min": 0, "max": 1},
    'balanced_accuracy': {
        'proba': 0, 'greater_is_better': 1, "min": 0, "max": 1},
    'cohen_kappa': {'proba': 0, 'greater_is_better': 1, "min": 0, "max": 1},
    'f1_score': {'proba': 0, 'greater_is_better': 1, "min": 0, "max": 1},
    'gini': {'proba': 1, 'greater_is_better': 1, "min": 0.5, "max": 1},
    'youden': {'proba': 0, 'greater_is_better': 1, "min": -1, "max": 1},
    'log_loss': {'proba': 1, 'greater_is_better': 0, "min": 0, "max": np.inf}
}


METRICS_REGRESSION = {
    'mean_absolute_error': {'greater_is_better': 0, "min": 0, "max": np.inf},
    'mean_squared_error': {'greater_is_better': 0, "min": 0, "max": np.inf},
    'median_absolute_error': {'greater_is_better': 0, "min": 0, "max": np.inf},
    'r2_score': {'greater_is_better': 1, "min": -np.inf, "max": 1},
    'mean_absolute_percentage_error': {
        'greater_is_better': 0, "min": 0, "max": np.inf},
    'mean_percentage_error': {'greater_is_better': 0, "min": 0, "max": np.inf},
    'mean_squared_percentage_error': {
        'greater_is_better': 0, "min": 0, "max": np.inf}
}


def _compute_binary_metric(metric, y_true, y_pred=None, y_pred_proba=None):
    """
    Compute metric for binary classification problem.

    Parameters
    ----------
    metric : str
        Binary classification metric. If metric is not valid, and execption is
        raised.

    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    y_pred_proba : array, shape [n_samples]
        Estimated probability of target=1 as returned by a classifier.
    """
    if metric == "auc_roc":
        metric_value = auc_roc(y_true, y_pred_proba)
    elif metric == "auc_pr":
        metric_value = auc_pr(y_true, y_pred_proba)
    elif metric == "balanced_accuracy":
        metric_value = balanced_accuracy(y_true, y_pred)
    elif metric == "cohen_kappa":
        metric_value = cohen_kappa_score(y_true, y_pred)
    elif metric == "gini":
        metric_value = gini(y_true, y_pred_proba)
    elif metric == "log_loss":
        metric_value = log_loss(y_true, y_pred_proba)
    elif metric == "f1_score":
        metric_value = f1_score(y_true, y_pred)
    elif metric == "youden":
        metric_value = youden_index(y_true, y_pred)

    metric_info = METRICS_BINARY_CLASSIFICATION[metric]

    if not metric_info['greater_is_better']:
        return -metric_value
    else:
        return metric_value


class ScorerBinaryClassification(Scorer):
    """
    Scorer for binary classification. List of available metrics:

    .. code::

        ["auc_roc", "auc_pr", "balanced_accuracy", "cohen_kappa", "f1_score",
         "gini", "log_loss", "youden"]

    Parameters
    ----------
    metrics : array-like, shape = [n_metrics]
        The name of the metrics to include in the scoring function.

    weights : array-like, shape = [n_metrics] (default=None)
        The weights associated with metrics. If not given, all metrics are
        supposed to have equal weight. If weights do not sum up 1, an exception
        is raised.

    See also
    --------
    grmlab.modelling.metrics
    """
    def score(self, model, X, y):
        """
        Compute scoring function with binary classification metrics.

        Parameters
        ----------
        model : object
            A model class instance having ``GRMlabModel`` as a parent class.
            This model must have been fitted, otherwise an exception is raised.

        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        See also
        --------
        grmlab.modelling.base.GRMlabModel
        """
        # Check whether model is a GRMlabModel
        if not isinstance(model, GRMlabModel):
            raise TypeError("model {} does not have GRMlabModel as a parent "
                            "class.".format(model.__class__.__name__))

        self._check_metrics()

        if self._is_pred_required:
            y_pred = model.predict(X)
        else:
            y_pred = None

        if self._is_pred_proba_required:
            y_pred_proba = model.predict_proba(X)[:, 1]
        else:
            y_pred_proba = None

        scores = np.asarray([_compute_binary_metric(metric, y, y_pred,
                            y_pred_proba) for metric in self.metrics])

        if self.weights is not None:
            return np.dot(scores, np.asarray(self.weights))
        else:
            return scores.sum()

    def _check_metrics(self):
        """"""
        for metric in self.metrics:
            list_metrics = list(METRICS_BINARY_CLASSIFICATION.keys())
            if metric not in list_metrics:
                raise ValueError("metric {} is not available.".format(metric))

            if METRICS_BINARY_CLASSIFICATION[metric]['proba'] == 1:
                self._is_pred_proba_required = True
            elif METRICS_BINARY_CLASSIFICATION[metric]['proba'] == 0:
                self._is_pred_required = True


def _compute_regression_metric(metric, y_true, y_pred=None):
    """
    Compute metric for regression problems.

    Parameters
    ----------
    metric : str
        Regression metric. If metric is not valid, and execption is raised.

    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets.
    """
    if metric == "mean_absolute_error":
        metric_value = mean_absolute_error(y_true, y_pred)
    elif metric == "mean_squared_error":
        metric_value = mean_squared_error(y_true, y_pred)
    elif metric == "median_absolute_error":
        metric_value = median_absolute_error(y_true, y_pred)
    elif metric == "r2_score":
        metric_value = r2_score(y_true, y_pred)
    elif metric == "mean_absolute_percentage_error":
        metric_value = mean_absolute_percentage_error(y_true, y_pred)
    elif metric == "mean_percentage_error":
        metric_value = mean_percentage_error(y_true, y_pred)
    elif metric == "mean_squared_percentage_error":
        metric_value = mean_squared_percentage_error(y_true, y_pred)

    metric_info = METRICS_REGRESSION[metric]

    if not metric_info['greater_is_better']:
        return -metric_value
    else:
        return metric_value


class ScorerRegression(Scorer):
    """
    Scorer for regression. List of available metrics:

    .. code::

        ["mean_absolute_error", "mean_squared_error", "median_absolute_error",
         "r2_score", "mean_absolute_percentage_error", "mean_percentage_error",
         "mean_squared_percentage_error"]

    Parameters
    ----------
    metrics : array-like, shape = [n_metrics]
        The name of the metrics to include in the scoring function.

    weights : array-like, shape = [n_metrics] (default=None)
        The weights associated with metrics. If not given, all metrics are
        supposed to have equal weight. If weights do not sum up 1, an exception
        is raised.

    See also
    --------
    grmlab.modelling.metrics
    """
    def score(self, model, X, y):
        """
        Compute scoring function with regression metrics.

        Parameters
        ----------
        model : object
            A model class instance having ``GRMlabModel`` as a parent class.
            This model must have been fitted, otherwise an exception is raised.

        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        See also
        --------
        grmlab.modelling.base.GRMlabModel
        """
        # Check whether model is a GRMlabModel
        if not isinstance(model, GRMlabModel):
            raise TypeError("model {} does not have GRMlabModel as a parent "
                            "class.".format(model.__class__.__name__))

        self._check_metrics()

        if self._is_pred_required:
            y_pred = model.predict(X)

        scores = np.asarray([_compute_regression_metric(metric, y, y_pred)
                            for metric in self.metrics])

        if self.weights is not None:
            return np.dot(scores, np.asarray(self.weights))
        else:
            return scores.sum()

    def _check_metrics(self):
        """"""
        for metric in self.metrics:
            list_metrics = list(METRICS_REGRESSION.keys())
            if metric not in list_metrics:
                raise ValueError("metric {} is not available.".format(metric))

            self._is_pred_proba_required = False
            self._is_pred_required = True
