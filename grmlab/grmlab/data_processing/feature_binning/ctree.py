"""
Univariate Conditional Inference Tree (CTree) optimized implementation for the
optimal binning problem.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2018.

import time

import numpy as np
import pandas as pd
import scipy.special as sp
import scipy.stats as st

from ..._lib.cppgrmlab.cppgrmlab import pearsonr_ctree_sort_b
from ..._lib.cppgrmlab.cppgrmlab import pearsonr_ctree_sort_d
from ..._lib.cppgrmlab.cppgrmlab import pearsonr_d_d
from ..._lib.cppgrmlab.cppgrmlab import pearsonr_d_i
from ..._lib.cppgrmlab.cppgrmlab import pearsonr_i_i
from ..._lib.cppgrmlab.cppgrmlab import unique_i
from ..._lib.cppgrmlab.util import grmlab_int_type
from ...core.base import GRMlabBase
from ...core.dtypes import is_binary
from ...core.exceptions import NotFittedException
from .util import categorical_others_group
from .util import process_data


DTYPES = ["categorical", "nominal", "numerical", "ordinal"]
DTYPE_CATNOM = ["categorical", "nominal"]
SPLIT_METHODS = ["entropy+k-tile", "gaussian", "k-tile"]


class CTree(GRMlabBase):
    """
    A univariate conditional inference tree.

    Parameters
    ----------
    dtype : str (default="numerical")
        The variable data type. Four dtypes supported: "categorical",
        "nominal", "numerical" and "ordinal".

    min_criterion : float (default=0.95)
        The value of the test statistic or 1 - (alpha or significance level)
        that must be exceeded in order to add a split.

    min_samples_split : int (default=2)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int (default=2)
        The minimum number of samples required to be at a leaf node.

    max_candidates : int (default=64)
        The maximum number of split points to perform an exhaustive search.

    dynamic_split_method : str (default="k-tile")
        The method to generate dynamic split points. Supported methods are
        “gaussian” for the Gaussian approximation, “k-tile” for the quantile
        approach and “entropy+k-tile” for a heuristic using class entropy. The
        "entropy+k-tile" method is only applicable when target is binary,
        otherwise, method "k-tile" is used instead.

    others_group : boolean (default=False)
        Whether to create an extra group with those values (categories) do not
        sufficiently representative. This option is available for dtypes
        "categorical" and "nominal".

    others_threshold : float (default=0.01)
        Merge categories which percentage of cases is below threshold to create
        an extra group. This option is available for dtypes "categorical" and
        "nominal".

    presort : boolean (default=True)
        Whether to presort the data to speed up the finding of best splits in
        fitting. This significantly speed up the fitting process.

    special_values : list or None (default=None)
        List of special values to be considered.

    verbose: int or boolean (default=False)
        Controls verbosity of output.

    Notes
    -----
    CTree is a non-parametric class of regression trees embedding
    tree-structured regression models into a well defined theory of conditional
    inference procedures :cite:`Hothorn_2006`. CTree uses a significant test
    procedure in order to select variables instead of selecting the variables
    that maximizes an information measure (Gini-coefficient, for instance).
    A new split is implemented when the criterion exceeds the value given by
    ``min_criterion``. This statistical approach ensures that the right-sized
    tree is grown without additional pruning or cross-validation, procedure
    typically required for other regression trees.

    The default value for the parameter ``max_candidates`` leads the
    performance of the algorithm. The algorithm do not perform an exact search
    but rather generates a few candidates using dynamic split methods
    :cite:`Chickering_2001`. To reduce the computation time for very large
    datasets set a smaller value.

    Example
    -------
    >>> from grmlab.data_processing.feature_binning import CTree
    >>> from sklearn.datasets import make_classification
    >>> X, y =  make_classification(n_samples=1000000, n_features=2,
    n_informative=2, n_redundant=0)
    >>> x = X[:, 1]
    >>> ct = CTree()
    >>> ct.fit(x, y)
    >>> ct.splits
    array([-0.73089634, -0.50799316, -0.27778961, -0.12926494,  0.06595937,
    0.45123609,  0.58590783])
    >>> ct.stats()
    CTree statistics:
    --------------------------------------
    total cpu time   := 0.1631
    presort time     := 0.1088 (66.70%)
    statistic time   := 0.0081 ( 4.97%)
    pearsonr time    := 0.0057 ( 3.53%)
    entropy time     := 0.0000 ( 0.00%)
    --------------------------------------
    """
    def __init__(self, dtype="numerical", min_criterion=0.95,
                 min_samples_split=2, min_samples_leaf=2, max_candidates=64,
                 dynamic_split_method="k-tile", others_group=True,
                 others_threshold=0.01, presort=True, special_values=None,
                 verbose=False):

        self.dtype = dtype
        self.min_criterion = min_criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_candidates = max_candidates
        self.dynamic_split_method = dynamic_split_method
        self.others_group = others_group
        self.others_threshold = others_threshold
        self.presort = presort
        self.special_values = special_values
        self.verbose = verbose

        # auxiliary
        self._target_binary = None
        self._n_split = None
        self._step_ktile = None
        self._splits = []
        self._categories = []
        self._others = []

        # timing
        self._time_total = None
        self._time_presort = 0
        self._time_categorical_transform = 0
        self._time_stats = 0
        self._time_pearson_corr = 0
        self._time_entropy = 0

        # flag
        self._is_fitted = False

    def fit(self, x, y, check_input=True):
        """
        Build a univariate CTree from the training set (x, y).

        Parameters
        ----------
        x : array-like, shape = [n_samples]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        check_input : boolean (default=True)
            Option to perform several input checking.

        Returns
        -------
        self : object
        """
        if check_input:
            if not isinstance(x, np.ndarray):
                raise TypeError("x must be a numpy array.")

            if not isinstance(y, np.ndarray):
                raise TypeError("y must be a numpy array.")

            if not x.size:
                raise ValueError("x cannot be empty.")

            if not y.size:
                raise ValueError("y cannot be empty.")

            if len(x) != len(y):
                raise ValueError("x and y must have the same length.")

        if self.dtype not in DTYPES:
            raise ValueError("dtype {} is not supported.".format(self.dtype))

        if self.min_criterion < 0.5 or self.min_criterion > 1.0:
            raise ValueError("min_criterion must be a float in [0.5, 1.0).")

        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2; "
                             "got {}.".format(self.min_samples_split))

        if self.min_samples_leaf < 2:
            raise ValueError("min_samples_leaf must be an integer >= 2; "
                             "got {}.".format(self.min_samples_leaf))

        if self.max_candidates < 2:
            raise ValueError("max_candidates must be an integer >= 2; "
                             "got {}.".format(self.max_candidates))

        if self.dynamic_split_method not in SPLIT_METHODS:
            raise ValueError("dynamic_split_method {} not supported."
                             .format(self.dynamic_split_method))

        # dynamic split points pre-computation
        if self.dtype not in DTYPE_CATNOM:
            if self.dynamic_split_method == "gaussian":
                self._n_split = self.max_candidates + 1
            else:
                self._step_ktile = 100.0 / self.max_candidates

        # modify options for categorical / nominal dtypes
        if self.dtype in DTYPE_CATNOM:
            self.presort = True
            self.max_candidates = None

        # fit ctree
        return self._fit(x, y)

    def stats(self):
        """CTree timing statistics."""
        if not self._is_fitted:
            raise NotFittedException(self)

        # timing
        per_presort = self._time_presort / self._time_total
        per_t_stats = self._time_stats / self._time_total
        per_t_corr = self._time_pearson_corr / self._time_total

        if self.dtype in DTYPE_CATNOM:
            per_t_cat = self._time_categorical_transform / self._time_total

            stats_str = (
                "\nCTree statistics:                   \n"
                "--------------------------------------\n"
                "total cpu time   := {:.4f}            \n"
                "presort time     := {:.4f} ({:6.2%}) \n"
                "statistic time   := {:.4f} ({:6.2%}) \n"
                "pearsonr time    := {:.4f} ({:6.2%}) \n"
                "categorical time := {:.4f} ({:6.2%}) \n"
                "--------------------------------------\n"
                ).format(
                self._time_total, self._time_presort, per_presort,
                self._time_stats, per_t_stats, self._time_pearson_corr,
                per_t_corr, self._time_categorical_transform, per_t_cat)
        else:
            per_entropy = self._time_entropy / self._time_total

            stats_str = (
                "\nCTree statistics:                   \n"
                "--------------------------------------\n"
                "total cpu time   := {:.4f}            \n"
                "presort time     := {:.4f} ({:6.2%}) \n"
                "statistic time   := {:.4f} ({:6.2%}) \n"
                "pearsonr time    := {:.4f} ({:6.2%}) \n"
                "entropy time     := {:.4f} ({:6.2%}) \n"
                "--------------------------------------\n"
                ).format(
                self._time_total, self._time_presort, per_presort,
                self._time_stats, per_t_stats, self._time_pearson_corr,
                per_t_corr, self._time_entropy, per_entropy)

        print(stats_str)

    def _find_split(self, unique_values, unique_target, n_uniq_vals, values,
                    target, n_event):

        binary_entropy = (self._target_binary and
                          self.dynamic_split_method == 'entropy+k-tile')
        if self.presort and binary_entropy:
            start = time.perf_counter()
            c = (values[:-1]+values[1:])/2
            split_candidates = np.unique(c[np.ediff1d(target) != 0])
            n_uniq_vals = len(split_candidates)
            self._time_entropy += time.perf_counter() - start
        else:
            split_candidates = unique_values[:-1]

        if self.dtype in DTYPE_CATNOM:
            if self._target_binary:
                split_candidates = unique_i(values[target == 1])
        elif n_uniq_vals - 1 > self.max_candidates:
            split_candidates = self._generate_candidates(split_candidates)

        max_criterion = 0.0
        best_split = None

        for c in split_candidates:
            if self.presort:
                pivot = np.searchsorted(values, c, side='right')

                # stopping criterion
                stop0 = (len(values) - pivot >= self.min_samples_leaf)
                stop1 = (pivot >= self.min_samples_leaf and stop0)
                pivot2 = np.searchsorted(unique_values, c, side='right')
                stop2 = (len(unique_values[:pivot2]) > 1)
                stop3 = (len(unique_target[:pivot2]) > 1)

                if ((stop1 and stop2) and stop3):
                    stat = np.abs(self._pearsonr_sort(target, pivot, n_event))
                    if stat > max_criterion:
                        max_criterion = stat
                        best_split = c
            else:
                subset = (values <= c)
                split_index = np.where(subset)[0]

                # stopping criterion
                stop0 = len(values) - len(split_index) >= self.min_samples_leaf
                stop1 = (len(split_index) >= self.min_samples_leaf and stop0)
                stop2 = (len(np.unique(target[split_index])) > 1)
                stop3 = (len(np.unique(values[split_index])) > 1)

                if ((stop1 and stop2) and stop3):
                    stat = np.abs(self._pearsonr(subset, target))
                    if stat > max_criterion:
                        max_criterion = stat
                        best_split = c

        return best_split

    def _fit(self, x, y):
        """Build conditional inference tree recursively."""
        init_ctree = time.perf_counter()

        values = x.copy()
        target = y.copy()

        # return values and target without missing and special values
        values, target, _ = process_data(values, target, None, self.special_values)

        # check if binary target to allow optimizations
        self._target_binary = is_binary(target)

        # apply categorical/nominal transformation
        if self.dtype in DTYPE_CATNOM:
            init_cat = time.perf_counter()

            if self.others_group:
                # extract others group if occurrences below threshold
                values, target, _, self._others = categorical_others_group(
                    values, target, None, self.others_threshold, self.verbose)

                # if no other group was created update others_group to avoid
                # creating of empty group
                if len(self._others) == 0:
                    self.others_group = False

            # apply categorical/nominal to ordinal transformation
            self._categories, values = self._categorical_transform(
                values, target)

            self._time_categorical_transform = time.perf_counter() - init_cat

        # presort values and target to trigger algorithmic optimizations
        if self.presort:
            init_presort = time.perf_counter()
            idx = np.argsort(values, kind='quicksort')
            values = values[idx]
            target = target[idx]
            self._time_presort = time.perf_counter() - init_presort

        # start recursive tree generation
        self._recurse(values, target, id=0)
        self._time_total = time.perf_counter() - init_ctree

        # Fit completed successfully
        self._is_fitted = True

        return self

    def _generate_candidates(self, split_candidates):
        if self.dynamic_split_method == "gaussian":
            max_c = np.max(split_candidates)
            mu = np.mean(split_candidates)
            sd = np.std(split_candidates)
            split_candidates = []
            for i in range(1, self.max_candidates):
                c = mu + sd * st.norm.ppf(i / self._n_split)
                if c < max_c:
                    split_candidates.append(c)
        else:
            bands = np.unique(
                np.percentile(split_candidates,
                              [i*self._step_ktile
                               for i in range(self.max_candidates)]))
            split_candidates = bands[1:len(bands)-1]

        return split_candidates

    def _pearsonr(self, x, y):
        """Compute Pearson's correlation using cppgrmlab functions."""
        start = time.perf_counter()
        if self._target_binary:
            r = pearsonr_i_i(x, y)
        else:
            r = pearsonr_d_d(x, y)
        self._time_pearson_corr += time.perf_counter() - start

        return r

    def _pearsonr_sort(self, y, pivot, n_event):
        start = time.perf_counter()
        if self._target_binary:
            r = pearsonr_ctree_sort_b(y, pivot, n_event)
        else:
            r = pearsonr_ctree_sort_d(y, pivot)
        self._time_pearson_corr += time.perf_counter() - start

        return r

    def _recurse(self, values, target, id):
        # compute pearsonr's statistic and pvalue
        stat, pvalue = self._statistic(values, target)
        stat = abs(stat)

        # re-use unique values
        if self.dtype in DTYPE_CATNOM:
            unique_vals = unique_i(values)
        else:
            unique_vals = np.unique(values)
        n_uniq_vals = len(unique_vals)

        # re-use unique target
        if self._target_binary:
            counts = np.bincount(target.astype(grmlab_int_type))
            n_uniq_targ = len(counts)
            unique_targ = range(n_uniq_targ)
        else:
            unique_targ = np.unique(target)
            n_uniq_targ = len(unique_targ)

        # check whether stop partitioning
        terminate = self._terminate_recursion(n_uniq_vals, n_uniq_targ, pvalue)

        if not terminate:
            n_event = counts[1] if self._target_binary else 0

            # find best split point for a given array
            split = self._find_split(unique_vals, unique_targ, n_uniq_vals,
                                     values, target, n_event)

            if split:
                self._splits.append(split)
                if self.presort:
                    # find first value greater than split (pivot)
                    # this makes a huge difference for large arrays
                    pivot = np.searchsorted(values, split, side='right')
                    left_idx = np.arange(pivot)
                    right_idx = np.arange(pivot, len(values))
                else:
                    left_idx = np.where(values <= split)[0]
                    right_idx = np.where(values > split)[0]

                self._recurse(values[left_idx], target[left_idx], id+1)
                self._recurse(values[right_idx], target[right_idx], id+2)

    def _statistic(self, values, target):
        """Compute Pearson's statistic and p-value."""
        start = time.perf_counter()
        if self._target_binary:
            r = pearsonr_d_i(values, target)
        else:
            r = pearsonr_d_d(values, target)

        r = max(min(r, 1.0), -1.0)
        df = len(values) - 2
        if np.abs(r) == 1.0:
            pval = 1.0
        else:
            try:
                tt = r*r * (df / ((1.0 - r) * (1.0 + r)))
                pval = 1.0 - sp.betainc(0.5*df, 0.5, df/(df+tt))
            except Exception:
                pval = np.nan

        self._time_stats += time.perf_counter() - start

        return r, pval

    def _terminate_recursion(self, n_values, n_target, pvalue):
        stop1 = (n_values > self.min_samples_split)
        stop2 = (n_target >= 2)
        splittable = stop1 and stop2
        return (np.abs(pvalue) < self.min_criterion) or not splittable

    def _categorical_transform(self, x, y):
        # 1 compute WoE
        n = len(x)
        # cat_unique, records = np.unique(x, return_counts=True)
        if self.dtype == "categorical":
            x = x.astype(str)

        cat_unique, records = self._unique_count(x)

        self.max_candidates = len(cat_unique)

        if self.max_candidates < 2:
            raise ValueError("max_candidates must be >= 2.")

        if self._target_binary:
            if self.dtype == "nominal":
                x0 = x[(y == 0)]
                x1 = x[(y == 1)]
            else:
                d = dict(map(reversed, enumerate(cat_unique)))
                xx = np.fromiter(map(d.get, x), dtype=grmlab_int_type, count=n)
                x0 = xx[(y == 0)]
                x1 = xx[(y == 1)]

            # nonevent_cat, nonevent = np.unique(x0, return_counts=True)
            nonevent_cat, nonevent = self._unique_count(x0)

            if nonevent.size == self.max_candidates:
                event = records - nonevent
            else:
                lst_event = []
                lst_nonevent = []
                if self.dtype == "nominal":
                    for i, cat in enumerate(cat_unique):
                        if cat in nonevent_cat:
                            ne = nonevent[np.where(nonevent_cat == cat)][0]
                            lst_event.append(records[i] - ne)
                            lst_nonevent.append(ne)
                        else:
                            lst_event.append(records[i])
                            lst_nonevent.append(0)
                else:
                    for i in range(self.max_candidates):
                        if i in nonevent_cat:
                            ne = nonevent[np.where(nonevent_cat == i)][0]
                            lst_event.append(records[i] - ne)
                            lst_nonevent.append(ne)
                        else:
                            lst_event.append(records[i])
                            lst_nonevent.append(0)
                event = np.asarray(lst_event)
                nonevent = np.asarray(lst_nonevent)

            eps = 1e-15
            cat_woe = np.log((eps + nonevent*len(x1)) / (eps + event*len(x0)))
            idx = np.argsort(cat_woe)[::-1][:self.max_candidates]
        else:
            cat_mean = [np.mean(y[x == u]) for u in cat_unique]
            idx = np.argsort(cat_mean)[::-1][:self.max_candidates]

        # 2. order array descending WoE and map
        cat_unique = np.array(cat_unique)[idx]
        d = dict(map(reversed, enumerate(cat_unique)))
        x = np.fromiter(map(d.get, x), dtype=grmlab_int_type, count=len(x))

        return cat_unique, x

    @staticmethod
    def _unique_count(x):
        uniq_s = pd.value_counts(x)
        uniq_c = uniq_s.index.values
        idx = np.argsort(uniq_c)
        return uniq_c[idx], uniq_s.values[idx]

    @property
    def splits(self):
        """
        CTree split points.

        Returns
        -------
        splits : numpy.ndarray
        """
        if not self._is_fitted:
            raise NotFittedException(self)

        _splits_points = sorted(self._splits)
        if self.dtype in DTYPE_CATNOM:
            cat = [-1] + list(_splits_points) + [len(self._categories)-1]

            cat_splits = []
            for i in range(len(cat)-1):
                cat_splits.append(
                    self._categories[range(cat[i]+1, cat[i+1]+1)])

            if self.others_group and len(self._others):
                # add others group
                cat_splits.append(self._others)

            return np.asarray(cat_splits)
        else:
            return np.asarray(_splits_points)
