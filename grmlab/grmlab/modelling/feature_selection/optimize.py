"""
Feature selection optimization algorithm.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import numbers

import numpy as np

from sklearn.base import MetaEstimatorMixin
from sklearn.feature_selection.base import SelectorMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y

from ...core.base import GRMlabBase
from ...core.exceptions import NotFittedException
from .optimization import FeatureSelectionSolver


class SelectOptimize(GRMlabBase, MetaEstimatorMixin, SelectorMixin):
    """
    Feature selection optimization.

    Select the optimal subset of features satisfying constraints.
    The IP formulation aims to maximize the linear correlation statistic.
    This type of objective functions has been successfully applied in the
    context of robust logistic regression algorithms :cite:`Feng_2014`.

    Parameters
    ----------
    method : str, 'classification'
        Type of problem. Currently only support for 'classification'.

    feature_names : array, shape (n_features, ) or None (default=None)
        The name of the features. If None, use integer indices in constraints
        excluded, fixed and groups. If array, the names of features in
        constraints is required.

    n_min_features : int or None (default=None)
        The minimum number of features to select. If None, at least one feature
        is selected.

    n_max_features : int or None (default=None)
        The maximum number of features to select. If None, half of the features
        are selected.

    max_correlation : float (default=0.6)
        The maximum positive correlation among selected features.

    max_correlation_target : float (default=0.3):
        The maximum absolute correlation between features and target.

    abs_correlation : int or boolean (default=True)
        Whether to consider the maximum or the absolute maximum correlation.

    excluded : array, shape (n_excluded, ) (default=[])
        If feature_names=None, the indices of the features to exclude.
        Otherwise, the name of the features to be excluded.

    fixed : array, shape (n_fixed, ) (default=[])
        If feature_names=None, the indeces of the features to fix.
        Otherwise, the name of the features to be fixed. The corresponding
        features will be included if a feasible solution exists.

    groups : array, shape (n_groups, ) (default=[])
        The group constraints constructed as a tuple
        ``([indices], n_min, n_max)``.

        ``[indices]`` : if feature_names=None, the array of features indices
        belonging to a group. Otherwise, the array of feature names.
        ``n_min`` : the minimum number of features to select from a group
        ``n_max`` : the maximum number of features to select from a group

    verbose : int or boolean (default=False)
        Controls verbosity of output.

    Attributes
    ----------
    n_features_ : int
        The number of selected features.

    support_ : array of shape [n_features]
        The mask of selected features.
    """
    def __init__(self, method, feature_names=None, n_min_features=None,
                 n_max_features=None, max_correlation=0.6,
                 max_correlation_target=0.3, abs_correlation=True,
                 excluded=None, fixed=None, groups=None, verbose=False):

        self.method = method
        self.feature_names = feature_names
        self.n_min_features = n_min_features
        self.n_max_features = n_max_features
        self.max_correlation = max_correlation
        self.max_correlation_target = max_correlation_target
        self.abs_correlation = abs_correlation
        self.excluded = excluded if excluded is not None else []
        self.fixed = fixed if fixed is not None else []
        self.groups = groups if groups is not None else []
        self.verbose = verbose

        # auxiliary
        self._solver = None

        # attributes
        self.n_features_ = None
        self.support_ = None

        # flag
        self._is_fitted = False

    def fit(self, X, y):
        """
        Fit MIP optimization problem and solve.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
        """
        return self._fit(X, y)

    def stats(self):
        """Internal mixed-integer programming solver statistics."""
        if not self._is_fitted:
            raise NotFittedException(self)

        return self._solver.summary_statistics()

    def _check_constraints_and_indices(self, n_features):
        """Check whether constraints indices are valid."""
        if self.feature_names is not None:
            # excluded constraint
            excluded = [self._find_feature_name_index(feature)
                        for feature in self.excluded]

            # fixed constraint
            fixed = [self._find_feature_name_index(feature)
                     for feature in self.fixed]

            # group constraints
            groups = [([self._find_feature_name_index(feature)
                        for feature in group[0]], group[1], group[2])
                      for group in self.groups]
        else:
            # excluded constraint
            excluded = [self._check_index_within_bounds(
                "excluded", feature_id, n_features)
                for feature_id in self.excluded]

            # fixed constraint
            fixed = [self._check_index_within_bounds(
                "fixed", feature_id, n_features)
                for feature_id in self.fixed]

            # group constraints
            groups = [([self._check_index_within_bounds(
                "group_{0}".format(grp_id), feature_id, n_features)
                for feature_id in group[0]], group[1], group[2])
                for grp_id, group in enumerate(self.groups)]

        return excluded, fixed, groups

    def _check_index_within_bounds(self, constraint_type, index, n_features):
        if index < 0 or index >= n_features:
            raise IndexError("index {} in constraint {} out of bounds.".format(
                index, constraint_type))

        return index

    def _find_feature_name_index(self, feature_name):
        if feature_name not in self.feature_names:
            raise ValueError("feature {} not in feature_names".format(
                feature_name))

        return next(i for i, name in enumerate(self.feature_names) if
                    feature_name == name)

    def _fit(self, X, y):
        # Check parameters
        if self.method not in ("classification", "regression"):
            raise ValueError("method must be classification or regression.")

        if (self.feature_names is not None) and (
                not isinstance(self.feature_names, (np.ndarray, list))):
            raise TypeError("feature_names must be a numpy array or list.")

        if self.n_min_features is not None and self.n_min_features < 0:
            raise ValueError("n_min_features must be a positive integer.")

        if self.n_max_features is not None and self.n_max_features < 0:
            raise ValueError("n_max_features must be a positive integer.")

        if ((self.n_min_features is not None and
                self.n_max_features is not None)
                and self.n_min_features > self.n_max_features):
            raise ValueError("n_min_features must be <= n_max_features.")

        if not isinstance(self.max_correlation, numbers.Number) or (
                self.max_correlation < 0.0 or self.max_correlation > 1.0):
            raise ValueError("max_correlation must be a positive number "
                             "in [0, 1].".format(self.max_correlation))

        if not isinstance(self.max_correlation_target, numbers.Number) or (
                self.max_correlation_target < 0.0 or
                self.max_correlation_target > 1.0):
            raise ValueError("max_correlation_target must be a positive "
                             "number in [0, 1].".format(self.max_correlation))

        # check X, y consistency
        X, y = check_X_y(X, y, "csc", ensure_min_features=2)

        # Initialization
        n_features = X.shape[1]

        if self.n_min_features is None:
            n_min_features = 1
        else:
            n_min_features = self.n_min_features

        if self.n_max_features is None:
            n_max_features = n_features // 2
        else:
            n_max_features = self.n_max_features

        # Check constraints and return indices
        [excluded, fixed, groups] = self._check_constraints_and_indices(
            n_features)

        # Compute correlation matrix and select infeasible pairs
        if self.verbose:
            if self.abs_correlation:
                print("Compute correlation matrix. Maximum allowed "
                      "correlation among variables is set to {}."
                      .format(self.max_correlation))
            else:
                print("Compute correlation matrix. Maximum allowed absolute "
                      "correlation among variables is set to {}."
                      .format(self.max_correlation))

        corr = np.corrcoef(X.T, y)
        if self.abs_correlation:
            corr = np.absolute(corr)

        corr_X = corr[:n_features, :n_features]
        corr_X_y = corr[n_features, :-1]

        cases = list(zip(*np.where(corr_X >= self.max_correlation)))
        pairs = list(set(tuple(sorted(p)) for p in cases if p[0] != p[1]))

        excluded_target = list(*np.where(
            corr_X_y >= self.max_correlation_target))

        if self.verbose:
            print("Number of infeasible features pairs: {}".format(len(pairs)))
            print("Number of infeasible features (target): {}".format(
                len(excluded_target)))

        if self.method == "classification":
            # Compute coefficients c. Binarize multi-class target and
            # mask {1,-1} following one-vs-rest approach (OVR).
            lb = LabelBinarizer()
            Y = lb.fit_transform(y)
            mask = (Y == 1)
            Y_bin = np.ones(Y.shape, dtype=X.dtype)
            Y_bin[~mask] = -1
            c = np.dot(X.T, Y_bin).T.sum(axis=0)

            # Solve optimization problem
            self._solver = FeatureSelectionSolver(
                c=c, n_min_features=n_min_features,
                n_max_features=n_max_features, infeas_pairs=pairs,
                infeas_features=excluded_target, excluded=excluded,
                fixed=fixed, groups=groups, verbose=self.verbose)
        else:
            raise NotImplementedError

        self._solver.solve()

        self.support_ = self._solver._mask
        self.n_features_ = self._solver._n_selected_features

        # Fit completed successfully
        self._is_fitted = True

        return self

    def _get_support_mask(self):
        if not self._is_fitted:
            raise NotFittedException(self)

        return self._solver.get_support()
