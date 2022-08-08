"""
Constrained Recursive Feature Elimination.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import numbers

import numpy as np

from sklearn.base import clone
from sklearn.base import MetaEstimatorMixin
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target

from ...core.base import GRMlabBase
from ...core.exceptions import NotFittedException
from ..classification import GRMlabModelClassification
from ..model_optimization import Scorer
from ..model_optimization import ScorerBinaryClassification
from ..model_optimization import ScorerRegression
from ..regression import GRMlabModelRegression
from .optimization import FeatureSelectionSolver


class CRFE(GRMlabBase, MetaEstimatorMixin, SelectorMixin):
    """
    Constrained Recursive Feature Elimination (CRFE).

    CRFE is an extension of the RFE algorithm implemented in
    :cite:`scikit-learn`. See ``sklearn.feature_selection.RFE```.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.

    scorer: object
        A scorer class instance having ``Scorer`` as a parent class.

    cv : object
        A sklearn cross-validation iterator instance. Example:
        ``sklearn.model_selection.StratifiedKFold``.

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

    step : int or float, optional (default=1)
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.

    max_correlation : float (default=0.6)
        The maximum positive correlation among selected features.

    max_correlation_target : float or None (default=1.0):
        The maximum absolute correlation between features and target.

    abs_correlation : int or boolean (default=True)
        Whether to consider the maximum or the absolute maximum correlation.

    excluded : array, shape (n_excluded, ) or None (default=None)
        If feature_names=None, the indices of the features to exclude.
        Otherwise, the name of the features to be excluded.

    fixed : array, shape (n_fixed, ) or None (default=None)
        If feature_names=None, the indeces of the features to force.
        Otherwise, the name of the features to be fixed. The corresponding
        features will be included if a feasible solution exists.

    groups : array, shape (n_groups, ) or None (default=None)
        The group constraints constructed as a tuple
        ``([indices], n_min, n_max)``. ``[indices]`` : if feature_names=None,
        the array of features indices belonging to a group. Otherwise, the
        array of feature names.
        * ``n_min`` : the minimum number of features to select from a group.
        * ``n_max`` : the maximum number of features to select from a group.

    verbose : int or boolean (default=False)
        Controls verbosity of output.

    Attributes
    ----------
    n_features_ : int
        The number of selected features.

    support_ : array of shape [n_features]
        The mask of selected features.

    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1.

    estimator_ : object
        The external estimator fit on the reduced dataset.

    rfe_n_features_ : int
        The number of selected features after RFE.

    rfe_support_ : array of shape [n_features]
        The mask of selected features after RFE.

    rfe_ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1.

    rfe_estimator_ : object
        The external estimator fit on the reduced dataset after RFE.
    """
    def __init__(self, estimator, scorer=None, cv=None, feature_names=None,
                 n_min_features=None, n_max_features=None, step=1,
                 max_correlation=0.6, max_correlation_target=1.0,
                 abs_correlation=True, excluded=None, fixed=None, groups=None,
                 verbose=False):

        self.estimator = estimator
        self.scorer = scorer
        self.cv = cv
        self.feature_names = feature_names
        self.n_min_features = n_min_features
        self.n_max_features = n_max_features
        self.step = step
        self.max_correlation = max_correlation
        self.max_correlation_target = max_correlation_target
        self.abs_correlation = abs_correlation
        self.excluded = excluded if excluded is not None else []
        self.fixed = fixed if fixed is not None else []
        self.groups = groups if groups is not None else []
        self.verbose = verbose

        # attributes
        self.estimator_ = None
        self.n_features_ = None
        self.support_ = None

        # auxiliary
        self._iterations = []

        # best iteration results
        self._best_iter = None
        self._best_score = None

        # timing
        self._time_rfe = 0
        self._time_optimization = 0

        # flag
        self._is_fitted = False

    def fit(self, X, y):
        """
        Fit CRFE model und the underlying estimator on the selected features.

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

    def results(self):
        """
        Optimization results over iterations.

        Returns
        -------
        results : dict of lists
            A dict with keys as column headers and values as columns, that can
            be imported into a pandas DataFrame.
        """
        if not self._is_fitted:
            raise NotFittedException(self)

        header = ["n_features"]

        n_features = [it["opt_support"].sum() for it in self._iterations]

        data = [n_features]

        if self.scorer is not None:
            header += ["score"]

            score = [it["opt_score"] for it in self._iterations]
            data.append(score)

        if self.cv is not None:
            header += ["cv_mean_score", "cv_std_score", "cv_min_score",
                       "cv_max_score"]

            smean = [it["opt_score_stats"]["mean"] for it in self._iterations]
            sstd = [it["opt_score_stats"]["std"] for it in self._iterations]

            smin = [it["opt_score_stats"]["min"] for it in self._iterations]
            smax = [it["opt_score_stats"]["max"] for it in self._iterations]

            data.append(smean)
            data.append(sstd)
            data.append(smin)
            data.append(smax)

        return dict(zip(header, data))

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
        # check parameters
        if self.feature_names is not None:
            if not isinstance(self.feature_names, (np.ndarray, list)):
                raise TypeError("feature_names must be a numpy array or list.")

        if self.n_min_features is not None and self.n_min_features < 0:
            raise ValueError("n_min_features must be a positive integer.")

        if self.n_max_features is not None and self.n_max_features < 0:
            raise ValueError("n_max_features must be a positive integer.")

        if self.n_min_features is not None and self.n_max_features is not None:
            if self.n_min_features > self.n_max_features:
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

        if not isinstance(self.step, numbers.Number) or self.step <= 0:
            raise ValueError("Step must be a number > 0; got{}"
                             .format(self.step))

        # check scorer and target dtype
        self._target_dtype = type_of_target(y)

        if self.scorer is not None:
            if not isinstance(self.scorer, Scorer):
                raise TypeError("scorer {} does not have Scorer as parent "
                                "class.".format(
                                    self.scorer.__class__.__name__))

            if self._target_dtype in ("binary", "multiclass",
                                      "multilabel-indicator"):
                if not isinstance(self.scorer, ScorerBinaryClassification):
                    raise TypeError("scorer {} doest not have "
                                    "ScorerClassification as a parent Class."
                                    .format(self.scorer.__class__.__name__))
            else:
                if not isinstance(self.scorer, ScorerRegression):
                    raise TypeError("scorer {} doest not have "
                                    "ScorerRegression as a parent Class."
                                    .format(self.scorer.__class__.__name__))

        # check X, y consistency
        X, y = check_X_y(X, y, "csc", ensure_min_features=2)

        # initialization
        n_features = X.shape[1]

        if self.n_min_features is None:
            n_min_features = 1
        else:
            n_min_features = self.n_min_features

        if self.n_max_features is None:
            n_max_features = n_features // 2
        else:
            n_max_features = self.n_max_features

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)

        # check constraints and return indices
        [excluded, fixed, groups] = self._check_constraints_and_indices(
            n_features)

        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        # compute correlation matrix and select infeasible pairs
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

        # Elimination
        while np.sum(support_) > n_max_features:
            # Remaining features
            features = np.arange(n_features)[support_]

            # Rank the remaining features
            estimator = clone(self.estimator)

            if self.verbose:
                print("Fitting estimator with {} features."
                      .format(np.sum(support_)))

            estimator.fit(X[:, features], y)

            # Get coefficients and compute feature importances
            if hasattr(estimator, "coef_"):
                coefs = estimator.coef_
            elif hasattr(estimator, "feature_importances_"):
                coefs = estimator.feature_importances_
            else:
                raise RuntimeError("The estimator does not expose 'coef_' "
                                   "or 'feature_importances_' attributes.")

            if coefs.ndim > 1:
                importances = np.square(coefs).sum(axis=0)
                ranks = np.argsort(importances)
            else:
                importances = np.square(coefs)
                ranks = np.argsort(importances)

            # for sparse case is matrix
            ranks = np.ravel(ranks)

            # Solve MILP optimization and cross-validation
            c = np.zeros(n_features, dtype=np.float)
            c[support_] = -1 * importances
            self._solve_milp_cv(X, y, c, n_min_features, n_max_features, pairs,
                                excluded_target, excluded, fixed, groups)

            # Eliminate the worst features
            threshold = min(step, np.sum(support_) - n_max_features)

            # Solve elimination conflicts
            idx_eliminate = features[ranks][:threshold]
            idx_eliminate = [idx for idx in idx_eliminate if idx not in fixed]

            support_[idx_eliminate] = False
            ranking_[~support_] += 1

        # Set final attributes RFE
        self.rfe_support_ = support_
        self.rfe_ranking_ = ranking_
        self.rfe_n_features_ = support_.sum()

        features = np.arange(n_features)[support_]
        self.rfe_estimator_ = clone(self.estimator)
        self.rfe_estimator_.fit(X[:, features], y)

        # Set final attributes CRFE
        if self.scorer is not None:
            if self.cv is not None:
                best_iter = np.argmax(
                    [_iter["opt_score_stats"]["mean"]
                     for _iter in self._iterations])
            else:
                best_iter = np.argmax(
                    [_iter["opt_score"] for _iter in self._iterations])
        else:
            best_iter = -1

        self.support_ = self._iterations[best_iter]["opt_support"]
        self.n_features_ = self.support_.sum()

        # ranking
        ranking_ = np.ones(n_features, dtype=np.int)
        for iter in self._iterations:
            ranking_[~iter["opt_support"]] += 1

        self.ranking_ = ranking_

        features = np.arange(n_features)[self.support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y)

        # fit completed successfully
        self._is_fitted = True

        return self

    def _solve_milp_cv(self, X, y, c, n_min_features, n_max_features, pairs,
                       excluded_target, excluded, fixed, groups):
        """
        Solve MILP formulation to select the subset of features satisfying
        user constraints and maximizing total feature importances.
        """
        opt_solver = FeatureSelectionSolver(
            c=c, n_min_features=n_min_features, n_max_features=n_max_features,
            infeas_pairs=pairs, infeas_features=excluded_target,
            excluded=excluded, fixed=fixed, groups=groups,
            verbose=self.verbose)

        if self.verbose:
            print("Solving optimization problem.")

        opt_solver.solve()

        opt_obj = opt_solver._mipcl_obj
        opt_support = opt_solver.get_support()
        opt_n_features = opt_solver._n_selected_features
        opt_score = None
        opt_score_stats = None

        X_opt = X[:, opt_support]

        if self.verbose:
            print("Selected features after optimization: {}."
                  .format(opt_n_features))

        # Build GRMlabModel from estimator
        estimator = clone(self.estimator)
        if self._target_dtype in ("binary", "multiclass",
                                  "multilabel-indicator"):
            model = GRMlabModelClassification(name="", estimator=estimator)
        else:
            model = GRMlabModelRegression(name="", estimator=estimator)

        model.fit(X_opt, y)

        if self.scorer is not None:
            opt_score = self.scorer.score(model, X_opt, y)
            if self.verbose:
                print("Scoring value: {}.".format(opt_score))

        if self.cv is not None:
            scores = []

            for train_index, test_index in self.cv.split(X_opt, y):
                X_train, y_train = X_opt[train_index], y[train_index]
                X_test, y_test = X_opt[test_index], y[test_index]

                model.fit(X_train, y_train)
                scores.append(self.scorer.score(model, X_test, y_test))

            score_mean = np.mean(scores)
            score_max = np.max(scores)
            score_min = np.min(scores)
            score_std = np.std(scores)

            opt_score_stats = {
                "mean": score_mean,
                "std": score_std,
                "min": score_min,
                "max": score_max
            }

            if self.verbose:
                print("Scoring CV: min={}, mean={}, max={}."
                      .format(score_min, score_mean, score_max))

        information = {
            "opt_obj": opt_obj,
            "opt_support": opt_support,
            "opt_score": opt_score,
            "opt_score_stats": opt_score_stats
        }

        # Store subset information
        self._iterations.append(information)

    def _get_support_mask(self):
        if not self._is_fitted:
            raise NotFittedException(self)

        return self.support_

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def classes_(self):
        return self.estimator_.classes_
