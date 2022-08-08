"""
Utility functions data analysis module.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import numpy as np

from scipy.special import xlogy

from scipy.stats import distributions
from scipy.stats import rankdata
from scipy.stats import tiecorrect


def entropy(p):
    """
    Calculate entropy of an array of the probability distribution p.

    Parameters
    ----------
    p : {array-like}
        List of probabilities.
    """
    pk = np.asarray(p)
    return -xlogy(pk, pk).sum()


def js_divergence_multivariate(X, weights_option="uniform", normalize=True):
    """
    Calculate Jensen-Shannon divergence of a matrix of probability
    distributions.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_distributions]
        Matrix of probability distributions.

    weights_option : str (default="uniform")
        The type of weights selected for the probability distributions.

    normalize : int or boolean (default=True)
        Whether to normalize the divergence measure. Normalization is performed
        dividing by log(n_distributions), such that the value is in [0, 1].
    """
    if X.ndim == 2: 
        _, n = X.shape
        
        if weights_option == "uniform":
            weights = np.ones(n) / n
        elif weights_option == "exponential":
            weights = np.exp(-np.linspace(n, 0, n) / n)
            weights /= weights.sum()
        else:
            raise ValueError("weights_option {} not supported".format(
                weights_option))

        t1 = entropy(np.nansum(weights *  X, axis=1))
        t2 = np.nansum([weights[i] * entropy(X[:, i]) for i in range(n)])
        if normalize:
            return abs(t1 - t2) / np.log(n)
        else:
            return abs(t1 - t2)
    else:
        return np.nan


def kruskalwallis(*args):
    """Compute the Kruskal-Wallis test for (n>=2) independent samples."""
    args = list(map(np.asarray, args))
    num_groups = len(args)
    if num_groups < 2:
        raise ValueError("Need at least two groups")

    for arg in args:
        if arg.size == 0:
            return np.nan, np.nan

    n = np.asarray(list(map(len, args)))
    alldata = np.concatenate(args)
    ranked = rankdata(alldata)
    ties = tiecorrect(ranked)
    if ties == 0:
        return (np.nan, np.nan)
    else:
        # Compute sum^2/n for each group and sum
        j = np.insert(np.cumsum(n), 0, 0)
        ssbn = 0
        for i in range(num_groups):
            _sum = np.sum(ranked[j[i]:j[i+1]])
            ssbn += _sum * _sum / float(n[i])

        # totaln = np.sum(n, dtype=np.float64)  (stats.py)
        totaln = np.sum(n,dtype='uint64')
        h = 12.0 / (totaln * (totaln + 1)) * ssbn - 3 * (totaln + 1)
        df = num_groups - 1
        h /= ties

    return h, distributions.chi2.sf(h, df)


def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, numpy_style=True):
    """ Very close to numpy.percentile, but supports weights.

    Parameters
    ----------
    values : numpy.array
        Data.

    quantiles : array-like 
        Quantiles needed. Quantiles should be in [0, 100].

    sample_weight : array-like of the same length as `array` (default=None)

    values_sorted : Boolean (default=False)
        If True, then will avoid sorting of initial array.

    numpy_style : Boolean (default=True)
        If True, will correct output to be consistent with numpy.percentile.

    Return
    ------
    return : numpy.array
        Computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        return np.percentile(values, quantiles)
    sample_weight = np.array(sample_weight)
    if not (np.all(quantiles >= 0) and np.all(quantiles <= 100)):
        raise ValueError('quantiles should be in [0, 100]')

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if numpy_style:
        # To be consistent with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles/100, weighted_quantiles, values)


def weighted_value_counts(values, sample_weight=None):
    """ Count elements with the weights.

    Parameters
    ----------
    values : pd.Series
        Data.

    sample_weight : array-like of the same length as `array` (default=None)

    Return
    ------
    return : pd.Series
        unique categories and weighted count.
    """
    if sample_weight is None:
        return values.value_counts()
    name = values.name
    values = values.to_frame()
    values["weight"] = sample_weight
    return values.groupby(name).sum()["weight"].sort_values(ascending=False)
