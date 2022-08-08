"""
Minimum Description Length Principle (MDLP) algorithm.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2018.

import time

import numpy as np

from ..._lib.cppgrmlab.util import grmlab_int_type
from ...core.base import GRMlabBase
from ...core.exceptions import NotFittedException
from .util import process_data


SPLIT_METHODS = ["entropy+k-tile", "k-tile"]


class MDLP(GRMlabBase):
    """
    Minimum Description Length Principle (MDLP) for discretization of
    continuous variables with respect to a binary target.

    Parameters
    ----------
    min_samples_split : int (default=2)
        The value of the test statistic or 1 - (alpha or significance level)
        that must be exceeded in order to add a split.

    min_samples_leaf : int (default=2)
        The minimum number of samples required to be at a leaf node.

    max_candidates : int (default=64)
        The maximum number of split points to perform an exhaustive search.

    dynamic_split_method : str (default="k-tile")
        The method to generate dynamic split points. Supported methods are
        “k-tile” for the quantile approach and “entropy+k-tile” for a heuristic
        using class entropy.

    special_values : list or None (default=None)
        List of special vlaues to be considered.

    verbose: int or boolean (default=False)
        Controls verbosity of output.

    Notes
    -----
    The MDLP performs a discretization of continuos-valued features. MDLP is a
    univariate, supervised discretization which uses the entropy minimization
    heuristic for discretizing the range of continuos values into multiple
    intervals. For more details see :cite:`Fayyad_1993`.

    The default value for the parameter ``max_candidates`` leads the
    performance of the algorithm. The algorithm do not perform an exact search
    but rather generates a few candidates using dynamic split methods
    :cite:`Chickering_2001`. To reduce the computation time for very large
    datasets set a smaller value.

    Example
    -------
    >>> from grmlab.data_processing.feature_binning import MDLP
    >>> from sklearn.datasets import make_classification
    >>> X, y =  make_classification(n_samples=1000000, n_features=2,
    n_informative=2, n_redundant=0)
    >>> x = X[:, 1]
    >>> mdlp = MDLP()
    >>> mdlp.fit(x, y)
    >>> mdlp.splits
    [-2.2422369085221354, -2.0558994962983297, -1.9658495661757542,
    -1.8534808789728558, -1.779984492269751, -1.6840426363181984,
    -1.6407483166406007, -1.583271350999165, -1.5238862492735104,
    -1.436790371881143, -1.4168374930752337, -1.3144197018372763,
    -1.1934865593198198, -0.7103664790451043, -0.6955420857378173,
    -0.6515255549740968, -0.583771693003123, -0.5137029430993958,
    -0.46098417363091976, -0.11637014771025136, -0.11337266453438491,
    0.08376151989475386, 0.1907473541445114, 0.36529182279205324,
    1.4004101566758114, 2.037011272848645, 2.302174542377218,
    2.5209261448588527, 2.976899000842508, 3.214233268225657,
    3.923553488209723]
    >>> mdlp.stats()
    MDLP statistics:
    --------------------------------------
    total cpu time   := 2.2125
    presort time     := 0.1090 ( 4.93%)
    entropy time     := 0.3185 (14.40%)
    --------------------------------------
    """
    def __init__(self, min_samples_split=2, min_samples_leaf=2,
                 max_candidates=64, dynamic_split_method="k-tile",
                 special_values=None, verbose=False):

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_candidates = max_candidates
        self.dynamic_split_method = dynamic_split_method
        self.special_values = special_values
        self.verbose = verbose

        # auxiliary
        self._splits = []
        self._step_ktile = None

        # timing
        self._time_total = None
        self._time_presort = None
        self._time_entropy = 0

        # flag
        self._is_fitted = False

    def fit(self, x, y, check_input=True):
        """
        Build a univariate MDLP tree from the training set (x, y).

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

        if self.max_candidates < 2:
            raise ValueError("max_candidates must be an integer >= 2; "
                             "got {}.".format(self.max_candidates))

        if self.dynamic_split_method not in SPLIT_METHODS:
            raise ValueError("dynamic_split_method {} not supported."
                             .format(self.dynamic_split_method))

        # dynamic split points pre-computation
        self._step_ktile = 100.0 / self.max_candidates

        # fit MDPL tree
        return self._fit(x, y)

    def stats(self):
        """MDLP timing statistics."""
        if not self._is_fitted:
            raise NotFittedException(self)

        per_presort = self._time_presort / self._time_total
        per_entropy = self._time_entropy / self._time_total

        stats_str = (
            "\nMDLP statistics:                    \n"
            "--------------------------------------\n"
            "total cpu time   := {:.4f}            \n"
            "presort time     := {:.4f} ({:6.2%}) \n"
            "entropy time     := {:.4f} ({:6.2%}) \n"
            "--------------------------------------\n"
            ).format(self._time_total, self._time_presort, per_presort,
                     self._time_entropy, per_entropy)

        print(stats_str)

    def _entropy(self, s):
        start = time.perf_counter()
        n = len(s)
        ns1 = np.sum(s)
        ns0 = n - ns1
        pc1 = ns1 / n
        pc0 = ns0 / n
        ent = -(pc0 * np.log(pc0) + pc1 * np.log(pc1))
        self._time_entropy += time.perf_counter() - start
        return ent

    def _entropy_gain(self, y, s1, s2):
        n = len(y)
        ent_y = self._entropy(y)
        ent_s1 = len(s1) * self._entropy(s1)
        ent_s2 = len(s2) * self._entropy(s2)
        return ent_y - (ent_s1 + ent_s2) / n

    def _find_split(self, unique_values, unique_target, n_uniq_vals, values,
                    target):

        if self.dynamic_split_method == 'entropy+k-tile':
            # find boundary point that minimize the class information entropy
            # of the partition induced by each split.

            # slow method
            # split_candidates = np.unique([(values[i]+values[i+1])/2
            # for i in range(len(target)-1) if target[i] != target[i+1]])

            # fast method
            c = (values[:-1]+values[1:])/2
            split_candidates = np.unique(c[np.ediff1d(target) != 0])
            n_uniq_vals = len(split_candidates)
        else:
            split_candidates = unique_values[:-1]

        if n_uniq_vals - 1 > self.max_candidates:
            split_candidates = self._generate_candidates(split_candidates)

        max_entropy_gain = 0.0
        best_split = None

        for c in split_candidates:
            pivot = np.searchsorted(values, c, side='right')
            left_idx = np.arange(pivot)
            right_idx = np.arange(pivot, len(values))

            # stopping criterion
            stop0 = len(values) - pivot >= self.min_samples_leaf
            stop1 = pivot >= self.min_samples_leaf and stop0
            pivot2 = np.searchsorted(unique_values, c, side='right')
            stop2 = (len(unique_values[:pivot2]) > 1)
            stop3 = (len(unique_target[:pivot2]) > 1)

            if ((stop1 and stop2) and stop3):
                entropy = self._entropy_gain(target, target[left_idx],
                                             target[right_idx])
                if entropy > max_entropy_gain:
                    max_entropy_gain = entropy
                    best_split = c

        return best_split

    def _fit(self, x, y):
        """Build MDLP tree recursively."""
        init_mdlp = time.perf_counter()

        values = x.copy()
        target = y.copy()

        # return values and target without missing and special values
        values, target, _ = process_data(values, target, None, self.special_values)

        # presort values and target
        init_presort = time.perf_counter()
        idx = np.argsort(values, kind='quicksort')
        values = values[idx]
        target = target[idx]
        self._time_presort = time.perf_counter() - init_presort

        # start recursive tree generation
        self._recurse(values, target, id=0)
        self._time_total = time.perf_counter() - init_mdlp

        # fit completed successfully
        self._is_fitted = True

        return self

    def _generate_candidates(self, split_candidates):
        bands = np.unique(np.percentile(split_candidates,
                          [i*self._step_ktile
                           for i in range(self.max_candidates)]))
        split_candidates = bands[1:len(bands)-1]
        return split_candidates

    def _recurse(self, values, target, id):
        # re-use arrays
        unique_vals = np.unique(values)
        n_uniq_vals = len(unique_vals)

        counts = np.bincount(target.astype(grmlab_int_type))
        n_uniq_targ = len(counts)
        unique_targ = range(n_uniq_targ)

        split = self._find_split(unique_vals, unique_targ, n_uniq_vals, values,
                                 target)
        if split:
            self._splits.append(split)
            pivot = np.searchsorted(values, split, side='right')
            left_idx = np.arange(pivot)
            right_idx = np.arange(pivot, len(values))

            terminate = self._terminate_recursion(
                n_uniq_vals, n_uniq_targ, target, target[left_idx],
                target[right_idx])
            if not terminate:
                self._recurse(values[left_idx], target[left_idx], id+1)
                self._recurse(values[right_idx], target[right_idx], id+2)

    def _terminate_recursion(self, n_values, n_target, target, s1, s2):
        stop1 = (n_values > self.min_samples_split)
        stop2 = (n_target >= 2)
        splittable = stop1 and stop2

        n = len(target)
        ent_y = self._entropy(target)
        ent_s1 = self._entropy(s1)
        ent_s2 = self._entropy(s2)
        gain = ent_y - (len(s1) * ent_s1 + len(s2) * ent_s2) / n

        k = 2 if n - np.sum(target) > 0 else 1
        k1 = 2 if len(s1) - np.sum(s1) > 0 else 1
        k2 = 2 if len(s2) - np.sum(s2) > 0 else 1

        t0 = np.log(3**k - 2)
        t1 = k * ent_y
        t2 = k1 * ent_s1
        t3 = k2 * ent_s2
        delta = t0 - (t1 - t2 - t3)

        stop3 = gain <= (np.log2(n - 1) + delta) / n

        return stop3 or not splittable

    @property
    def splits(self):
        """
        MDLP split points.

        Returns
        -------
        splits : numpy.ndarray
        """
        return np.asarray(sorted(self._splits))
