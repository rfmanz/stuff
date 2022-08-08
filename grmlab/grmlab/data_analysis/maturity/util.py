"""
Maturity analysis utils.
"""

# Authors: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2020.

from sklearn.linear_model import LinearRegression

import numpy as np


def costFunction(x, y, sigma, beta, gamma, min_points):
    """Cost function of a segment"""

    if len(y) < min_points:
        return 9e999

    y_reg = y
    x_reg = x.reshape(-1, 1)

    reg = LinearRegression(fit_intercept=True).fit(x_reg, y_reg)

    residuals = y_reg - reg.predict(x_reg)
    D = durbin_watson(residuals)

    # Cost calculation and return
    return ((1/(sigma**2))*sum(np.power(residuals, 2)) +
            np.abs(D-2)**2 + beta + gamma*np.log(len(y)))


def durbin_watson(residuals):
    """Evaluation of the Durbin Watson statistic"""

    # if all residuals are 0
    if np.sum(np.power(residuals, 2)) < 1e-10:
        return 2

    return (sum((residuals[i+1]-residuals[i])**2
                for i in range(len(residuals)-1)) /
            np.sum(np.power(residuals, 2)))


def best_cost(x, y, cost, idx, sigma, beta, gamma, min_points):
    """Gets the best cost for a given split point (idx)"""

    candidates = enumerate(reversed(cost[0:idx]))
    return min(
        (c + costFunction(x[idx-k-1:idx], y[idx-k-1:idx],
                          sigma, beta, gamma, min_points),
         k+1) for k, c in candidates)


def infer_change_points(y, x=None, beta=None, gamma=0, min_points=3):
    """Calculates the split points of a piecewise function.

    It uses dynamic programming to minimize the cost-function.

    Parameters
    ----------
    y: list or numpy.array
        y values of the data

    x: list or numpy.array (default=None)
        x values of the data. If None, equal unit spaces will be given.

    beta: float (default=None)
        Cost of each split point. If None, 2*log(n) will be given.

    gamma: float (default=0)
        Segment length penalty.

    min_points: int (default=3)
        Minimum number of points in each segment.
    """

    if x is None:
        x = np.arange(0, len(y))
    if beta is None:
        beta = 2*np.log(len(y))

    # REVIEW if this can be used for unequal values of x
    sigma = np.median(np.abs(y - np.median(y)))
    if sigma < 1e-10:
        sigma = np.std(y)
    if sigma < 1e-10:
        # y is constant, no splits
        return [y], []

    # Best cost for each possible split point on y
    cost = [0]
    for idx in range(1, len(y)+1):
        c, k = best_cost(x, y, cost, idx, sigma, beta, gamma, min_points)
        cost.append(c)

    # Backtrack to recover the minimal-cost point splits.
    buckets = []
    split_points = []
    idx = len(y)
    while idx > 0:
        c, k = best_cost(x, y, cost, idx, sigma, beta, gamma, min_points)
        assert c == cost[idx]  # double check
        # append all the values in the same bucket
        buckets.append(y[idx-k:idx])
        idx -= k
        if idx != 0:
            split_points.append(idx)

    return list(reversed(buckets)), list(reversed(split_points))
