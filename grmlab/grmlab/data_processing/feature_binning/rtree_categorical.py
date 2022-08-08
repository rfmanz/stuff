"""
Univariate CART for categorical and nominal data.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import numpy as np
import pandas as pd

from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from ..._lib.cppgrmlab.util import grmlab_int_type
from ...core.base import GRMlabBase
from ...core.dtypes import is_binary
from ...core.exceptions import NotFittedException
from .util import categorical_others_group
from .util import process_data


class RTreeCategorical(GRMlabBase):
    """
    A univariate CART for categorical and nominal data.

    Parameters
    ----------
    dtype : str (default="categorical")
        The variable data type. Two dtypes supported: "categorical" and
        "nominal".

    min_samples_split : int (default=2)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int (default=2)
        The minimum number of samples required to be at a leaf node.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are
        defined as relative reduction in impurity. If None then unlimited
        number of leaf nodes.

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
    """
    def __init__(self, dtype="categorical", min_samples_split=2,
                 min_samples_leaf=2, max_leaf_nodes=None, others_group=True,
                 others_threshold=0.01, presort=True, special_values=None,
                 verbose=False):

        self.dtype = dtype
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.others_group = others_group
        self.others_threshold = others_threshold
        self.presort = presort
        self.special_values = special_values
        self.verbose = verbose

        # auxiliary
        self._target_binary = None
        self._n_split = None
        self._splits = []
        self._categories = []
        self._others = []

        # flag
        self._is_fitted = False

    def fit(self, x, y, sample_weight=None, check_input=False):
        """
        Build a univariate CART from the training set (x, y).

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

        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2; "
                             "got {}.".format(self.min_samples_split))

        if self.min_samples_leaf < 2:
            raise ValueError("min_samples_leaf must be an integer >= 2; "
                             "got {}.".format(self.min_samples_leaf))

        if self.max_leaf_nodes < 2:
            raise ValueError("max_leaf_nodes must be an integer >= 2; "
                             "got {}.".format(self.max_leaf_nodes))

        # fit rtree
        return self._fit(x, y, sample_weight)

    def _fit(self, x, y, sample_weight=None):
        """Build CART"""
        values = x.copy()
        target = y.copy()
        if sample_weight is not None:
            sample_weight = sample_weight.copy()

        # return values and target without missing and special values
        values, target, sample_weight = process_data(
            values, target, sample_weight, self.special_values)

        # check if binary target to allow optimizations
        self._target_binary = is_binary(target)

        if self.others_group:
            # extract others group if occurrences below threshold
            values, target, sample_weight, self._others = categorical_others_group(
                values, target, sample_weight, self.others_threshold,
                self.verbose)

            # if no other group was created update others_group to avoid
            # creating of empty group
            if len(self._others) == 0:
                self.others_group = False

        # apply categorical/nominal to ordinal transformation
        self._categories, values = self._categorical_transform(values, target)

        if self._target_binary:
            rtree = DecisionTreeClassifier(
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_leaf_nodes=self.max_leaf_nodes, presort=self.presort)
        else:
            rtree = DecisionTreeRegressor(
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_leaf_nodes=self.max_leaf_nodes, presort=self.presort)

        rtree.fit(values.reshape(-1, 1), target, sample_weight=sample_weight)

        # retrieve splits
        splits = np.unique(rtree.tree_.threshold)
        self._splits = splits[splits != _tree.TREE_UNDEFINED]

        # Fit completed successfully
        self._is_fitted = True

        return self

    def _categorical_transform(self, x, y):
        # 1 compute WoE
        n = len(x)
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

        _splits_points = sorted(np.floor(self._splits).astype(int))

        cat = [-1] + list(_splits_points) + [len(self._categories)-1]

        cat_splits = []
        for i in range(len(cat)-1):
            cat_splits.append(self._categories[range(cat[i]+1, cat[i+1]+1)])

        if self.others_group and len(self._others):
            # add others group
            cat_splits.append(self._others)

        return np.asarray(cat_splits)
