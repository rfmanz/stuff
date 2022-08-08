"""
Data preprocessing utils.
"""

# Authors: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
#          Carlos González Berrendero <c.gonzalez.berrender@bbva.com>
# BBVA - Copyright 2019.

import numpy as np

from ...core.dtypes import check_date_format


def next_month(date):
    """
    Returns next month.
    Parameters
    ----------
    date : integer date in the form yyyymm

    Returns
    -------
    next_month_date : integer date in the form yyyymm

    The following month of the input.
    """
    # December -> January
    if (date % 100 == 12):
        next_month_date = date - 11 + 100
    else:
        next_month_date = date + 1

    return next_month_date


def _change_date_format(date):
    """
    Changes date variable format.
        yyyymmdd -> yyymm

    Parameters
    ----------
    date: np.array

    """
    date_yyymm = np.zeros(len(date), dtype=np.int32)
    for i in range(len(date)):
        date_yyymm[i] = int(date[i]/100)
    return date_yyymm


def _populated_percentiles(y, n_bins=10):
    """
    Computes n equally weighted percentiles and chekc that all bins are
    populated.

    Parameters
    ----------
    y : list or numpy array
        Ground truth (correct) target values.

    n_bins : int (default=10)
        It defines the number of equal bins.

    Returns
    -------
    bins : list
        Array with the percentiles.
    """

    y = np.array(y)  # change to numpy array format
    # percentiles
    bins = np.percentile(y, np.arange(100/n_bins, 100, 100/n_bins))

    # check for empty bins
    population = []
    population.append(y[(y < bins[0])].shape[0])
    for i, elt in enumerate(bins[1:]):
        population.append(y[(y < elt) & (y >= bins[i])].shape[0])

    return bins[np.array(population) > 1]


def get_bins(y, bins=10):
    """
    Calculates bins based in an array.

    Parameters
    ----------
    y : list or numpy array

    bins : int, list or numpy array (default=10)
        If bins is an int, it defines the number of equal populated bins to
        discretize the values. If bins is a list or numpy array, it has to be
        1-dimensional and monotonic; it will discretize the values with this
        cuts, allowing non-uniform bin population.

    Returns
    -------
    n_bins : int
        Number of bins.

    bins : np.array
        Array with bin's limits.
    """
    if isinstance(bins, (int, np.int)):
        if bins == 1:
            return 1, [np.inf]
        bins = _populated_percentiles(y, bins)
        n_bins = bins.shape[0]+1  # number of bins
    elif isinstance(bins, (list, np.ndarray)):
        n_bins = len(bins)+1  # number of bins
        bins = bins
    else:
        raise TypeError("Parameter bins must be an integer, list or "
                        "numpy array")
    return n_bins, bins


def year_intervals(date_min, date_max):
    """
    Computes the a vector with non-disjunct year intervals
    from date_min to date_max.

    Parameters
    ----------
    date_min : integer date in the form yyyymm
        The first interval will be [date_min, date_min + 1 year]

    date_max : integer date in the form yyyymm
        The last interval will be [date_max - 1 year, date_max]

    Returns
    -------
    intervals : list of lists
        [[date_min, date_min + 1 year],
         [date_min+ 1 month, date_min + 1 month + 1 year],
          ...
         [date_max - 1 year, date_max]]
    """

    check_date_format(np.array([date_min, date_max]))
    intervals = []
    lim_inf = date_min
    lim_sup = date_min + 100
    date_max_loop = next_month(date_max)
    while (lim_sup <= date_max_loop):
        intervals.append([lim_inf, lim_sup])
        lim_inf = next_month(lim_inf)
        lim_sup = lim_inf + 100
    return intervals
