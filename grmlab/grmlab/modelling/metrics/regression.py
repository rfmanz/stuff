"""
Regression metrics.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import numpy as np
import pandas as pd

from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

from .util import _check_reg_targets


def zero_division(func):
    """wrapper for zero values"""
    def inner(*args, **kwargs):
        if "y_true" in kwargs:
            vec = kwargs["y_true"]
        else:
            vec = args[0]

        if sum(vec == 0) > 0:
            return 9999
        else:
            return func(*args, **kwargs)
    return inner


@zero_division
def mean_absolute_percentage_error(y_true, y_pred):
    r"""
    Compute the mean absolute percentage error (MAPE).

    .. math::

        MAPE = \frac{1}{n}\sum_{i=0}^n \left|\frac{y_i - \hat{y}_i}
        {y_i}\right|,

    where :math:`y` are the truth target values and :math:`\hat{y}` are the
    predicted target values.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated target values.

    Returns
    -------
    mape : float
    """
    y_type, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, None)

    if y_type == 'continuous-multioutput':
        raise ValueError("Multioutput not supported in "
                         "mean_absolute_percentage_error.")

    return np.mean(np.abs((y_true - y_pred) / y_true))


@zero_division
def mean_percentage_error(y_true, y_pred):
    r"""
    Compute the mean percentage error (MPE).

    .. math::

        MPE = \frac{1}{n}\sum_{i=0}^n \frac{y_i - \hat{y}_i}{y_i},

    where :math:`y` are the truth target values and :math:`\hat{y}` are the
    predicted target values.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated target values.

    Returns
    -------
    mpe : float
    """
    y_type, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, None)

    if y_type == 'continuous-multioutput':
        raise ValueError("Multioutput not supported in "
                         "mean_absolute_percentage_error.")

    return np.mean((y_true - y_pred) / y_true)


@zero_division
def median_percentage_error(y_true, y_pred):
    r"""
    Compute the median percentage error (MDPE).

    .. math::

        MDPE = \text{Median}\left(\sum_{i=0}^n \frac{y_i - \hat{y}_i}{y_i}\right),

    where :math:`y` are the truth target values and :math:`\hat{y}` are the
    predicted target values.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated target values.

    Returns
    -------
    mpe : float
    """
    y_type, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, None)

    if y_type == 'continuous-multioutput':
        raise ValueError("Multioutput not supported in "
                         "mean_absolute_percentage_error.")

    return np.median((y_true - y_pred) / y_true)


@zero_division
def std_percentage_error(y_true, y_pred):
    r"""
    Compute the standard deviation of percentage error.

    .. math::

        STDPE = \text{std}\left(\sum_{i=0}^n \frac{y_i - \hat{y}_i}{y_i}\right),

    where :math:`y` are the truth target values and :math:`\hat{y}` are the
    predicted target values.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated target values.

    Returns
    -------
    mpe : float
    """
    y_type, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, None)

    if y_type == 'continuous-multioutput':
        raise ValueError("Multioutput not supported in "
                         "mean_absolute_percentage_error.")

    return np.std((y_true - y_pred) / y_true)


def mean_error(y_true, y_pred):
    r"""
    Compute the mean error.

    .. math::

        ME = \frac{1}{n}\sum_{i=0}^n (y_i - \hat{y}_i),

    where :math:`y` are the truth target values and :math:`\hat{y}` are the
    predicted target values.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated target values.

    Returns
    -------
    mpe : float
    """
    y_type, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, None)

    if y_type == 'continuous-multioutput':
        raise ValueError("Multioutput not supported in "
                         "mean_absolute_percentage_error.")

    return np.mean(y_true - y_pred)


def median_error(y_true, y_pred):
    r"""
    Compute the median error.

    .. math::

        MDE = \text{Median}\left(\sum_{i=0}^n (y_i - \hat{y}_i)\right),

    where :math:`y` are the truth target values and :math:`\hat{y}` are the
    predicted target values.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated target values.

    Returns
    -------
    mpe : float
    """
    y_type, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, None)

    if y_type == 'continuous-multioutput':
        raise ValueError("Multioutput not supported in "
                         "mean_absolute_percentage_error.")

    return np.median(y_true - y_pred)


def std_error(y_true, y_pred):
    r"""
    Compute the std error.

    .. math::

        STDE = \text{std}\left(\sum_{i=0}^n (y_i - \hat{y}_i)\right),

    where :math:`y` are the truth target values and :math:`\hat{y}` are the
    predicted target values.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated target values.

    Returns
    -------
    mpe : float
    """
    y_type, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, None)

    if y_type == 'continuous-multioutput':
        raise ValueError("Multioutput not supported in "
                         "mean_absolute_percentage_error.")

    return np.std(y_true - y_pred)


@zero_division
def mean_squared_percentage_error(y_true, y_pred):
    r"""
    Compute the mean squared percentage error (MSPE).

    .. math::

        MSPE = \frac{1}{n}\sum_{i=0}^n \left(\frac{y_i - \hat{y}_i}
        {y_i}\right)^2,

    where :math:`y` are the truth target values and :math:`\hat{y}` are the
    predicted target values.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated target values.

    Returns
    -------
    mspe : float
    """
    y_type, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, None)

    if y_type == 'continuous-multioutput':
        raise ValueError("Multioutput not supported in "
                         "mean_absolute_percentage_error.")

    return np.mean(((y_true - y_pred) / y_true) ** 2)


_METRICS = ("mean_error", "median_error", "std_error",
            "mean_absolute_error", "mean_squared_error",
            "root_mean_squared_error",
            "median_absolute_error", "r2_score", "r2_adj_score",
            "mean_absolute_percentage_error", "mean_percentage_error",
            "median_percentage_error", "std_percentage_error",
            "mean_squared_percentage_error", "pearson_corr",
            "spearson_rank_corr", "kendall_rank_corr")

_METRICS_SHORTCUTS = ("me", "mde", "stde", "mae", "mse", "rmse", "mdae", "r2",
                      "mape", "mpe", "mdpe", "stdpe", "mspe", "pcc", "srcc",
                      "krcc")


def regression_report(y_true, y_pred, n_features=1, metrics=None,
                      output_dict=True):
    """
    Build a complete report with all or several metris for regression.

    Available metrics are::

        metrics = ["mean_error", "median_error", "std_error",
                   "mean_absolute_error", "mean_squared_error",
                   "root_mean_squared_error",
                   "median_absolute_error", "r2_score", "r2_adj_score",
                   "mean_absolute_percentage_error", "mean_percentage_error",
                   "median_percentage_error", "std_percentage_error",
                   "mean_squared_percentage_error", "pearson_corr",
                   "spearson_rank_corr", "kendall_rank_corr"]

    or their corresponding shortcuts::

        metrics = ["me", "mde", "stde", "mae", "mse", "rmse", "mdae", "r2",
                   "mape", "mpe", "mdpe", "stdpe", "mspe", "pcc", "srcc",
                   "krcc"]

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    n_features : int (default=1)
        Number of independent variables used to predict the target.

    metrics : array-like (default=None)
        Regression metrics for reporting. If None, all metrics are included.

    output_dict : bool (default=False)
        If True, return output as dict, otherwise return pandas.DataFrame.

    Returns
    -------
    report : dict or pandas.DataFrame
    """
    if metrics is not None:
        _metrics = [_check_metric(metric) for metric in metrics]
    else:
        _metrics = _METRICS

    y_type, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, None)
    n_samples = len(y_type)

    report_dict = {}

    for metric in _metrics:
        if metric == "mean_error":
            value = mean_error(y_true, y_pred)
        if metric == "median_error":
            value = median_error(y_true, y_pred)
        if metric == "std_error":
            value = std_error(y_true, y_pred)
        if metric == "mean_absolute_error":
            value = mean_absolute_error(y_true, y_pred)
        elif metric == "mean_squared_error":
            value = mean_squared_error(y_true, y_pred)
        elif metric == "root_mean_squared_error":
            value = np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == "median_absolute_error":
            value = median_absolute_error(y_true, y_pred)
        elif metric == "r2_score":
            value = r2_score(y_true, y_pred)
        elif metric == "r2_adj_score":
            value = 1-((1-n_samples)*(1-r2_score(y_true, y_pred)) /
                       (n_samples-n_features-1))
        elif metric == "mean_absolute_percentage_error":
            value = mean_absolute_percentage_error(y_true, y_pred)
        elif metric == "mean_percentage_error":
            value = mean_percentage_error(y_true, y_pred)
        elif metric == "median_percentage_error":
            value = median_percentage_error(y_true, y_pred)
        elif metric == "std_percentage_error":
            value = std_percentage_error(y_true, y_pred)
        elif metric == "mean_squared_percentage_error":
            value = mean_squared_percentage_error(y_true, y_pred)
        elif metric == "pearson_corr":
            value, p_value = pearsonr(y_true.flatten(), y_pred.flatten())
        elif metric == "spearson_rank_corr":
            value, p_value = spearmanr(y_true.flatten(), y_pred.flatten())
        elif metric == "kendall_rank_corr":
            value, p_value = kendalltau(y_true.flatten(), y_pred.flatten())

        report_dict[metric] = value

    if output_dict:
        return report_dict
    else:
        report_df = pd.DataFrame.from_dict(
            report_dict, orient='index').sort_index()

        return report_df


def _check_metric(metric):
    """Check if metric is valid."""
    if metric in _METRICS:
        return metric
    elif metric in _METRICS_SHORTCUTS:
        return next(_METRICS[i] for i, m in enumerate(_METRICS_SHORTCUTS)
                    if metric == m)
    else:
        raise ValueError("{} is not a valid imbalanced metric.".format(metric))
