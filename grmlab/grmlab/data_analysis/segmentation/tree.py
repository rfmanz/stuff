"""
Tree for segment calculation
"""

# Authors: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2020.

import numpy as np
import pandas as pd

from sklearn.utils import check_X_y

from ...core.base import GRMlabBase
from ...core.exceptions import NotFittedException
from .node import NodeTree
from .util import calc_iv
from .util import RANDOM_INT


def get_segments(X, y, split_vars=None, num_iter=100,
                 min_weight_fraction_leaf=0.1,
                 test_name="chow", alpha=0.05, max_depth=None,
                 random_state=1, user_splits=None, verbose=True):
    """
    Calculates the best segment for the database.

    The best segments are obtained from the tree with maximum IV. All the nodes
    from the tree have passed the test specified by the user.

    Parameters
    ----------
    X : pandas.DataFrame
        The train dataset.

    y : pandas.DataFrame
        The train target values.

    split_vars : list or None (default=None)
        List of variables to fit the tree. If None, all the variables in X
        dataframe will be used.

    num_iter : int (default=100)
        Number of trees initialized with different seed. The tree with higher
        metric is the one selected for the segmentation.

    min_weight_fraction_leaf : float (default=0.1)
        The minimum weighted fraction of the sum total of weights (of all the
        input samples) required to be at a leaf node. Samples have equal weight
        when sample_weight is not provided.

    test_name : str (default="chow")
        The name of the test. There are two options: The chow test ('chow') and
        the odds test ('odds')

    alpha : float (default=0.05)
        Significance level of the test.

    max_depth : int or None (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_weight_fraction_leaf samples.

    random_state : int (default=1)
        random_state is the seed used by the random number generator.

    user_splits : tuple, dict, None or False (default=None)
        If None, it will try the best split given by sklearn Tree.
        If False, this node will not try to split, it will be a leaf node.
        If tuple, it must have the structure:

            >>> user_splits = (feature, split_value)

        If dictionary, it must have the following structure:

            >>> user_splits = {"node": (feature, split_value) or None or False,
            >>>                "left": user_split_lelf,
            >>>                "right": user_split_right}

        At "left" or "right" keys any of the previous options can be set as
        value.

    """
    # check num_iter
    if isinstance(num_iter, int):
        if not (num_iter > 0):
            raise ValueError("num_iter must be a positive integer.")
    else:
        raise TypeError("num_iter must be integer.")

    # check random_state
    if isinstance(random_state, (int, np.int32, np.int64)):
        if not (random_state >= 0):
            raise ValueError("random_state must be a non negative integer.")
    else:
        raise TypeError("random_state must be integer.")

    # initialize random seeds
    np.random.seed(seed=random_state)
    random_array = np.random.randint(RANDOM_INT, size=num_iter)

    # initialize the best tree
    best_tree = Tree(split_vars=split_vars,
                     min_weight_fraction_leaf=min_weight_fraction_leaf,
                     test_name=test_name,
                     alpha=alpha, max_depth=max_depth,
                     random_state=random_array[0], user_splits=user_splits)
    best_tree.fit(X, y)

    iter_i = 0
    if verbose:
        print("     - iter " + str(iter_i) + ":", best_tree.metric, "B")

    # loop through all the trees
    for rand_ in random_array[1:]:
        iter_i += 1
        next_tree = Tree(split_vars=split_vars,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         test_name=test_name, alpha=alpha, max_depth=max_depth,
                         random_state=rand_, user_splits=user_splits)
        next_tree.fit(X, y)

        # select the best tree on each iteration
        if next_tree.metric > best_tree.metric:
            best_tree = next_tree
            if verbose:
                print("     - iter " + str(iter_i) + ":",
                      best_tree.metric, "B")
        else:
            if verbose:
                print("     - iter " + str(iter_i) + ":",
                      next_tree.metric)

    return best_tree


class Tree(GRMlabBase):
    """
    Tree for which all its nodes are testes among each other.

    Fits the data to a tree classifier. For each node split, the resultant leafs
    are tested to pass the specified test set by the user.

    Parameters
    ----------
    split_vars : list or None (default=None)
        List of variables to fit the tree. If None, all the variables in the
        dataframe will be used.

    min_weight_fraction_leaf : float (default=0.1)
        The minimum weighted fraction of the sum total of weights (of all the
        input samples) required to be at a leaf node. Samples have equal weight
        when sample_weight is not provided.

    test_name : str (default="chow")
        The name of the test. There are two options: The chow test ('chow') and
        the odds test ('odds')

    alpha : float (default=0.05)
        Significance level of the test.

    max_depth : int or None (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_weight_fraction_leaf samples.

    random_state : int (default=1)
        random_state is the seed used by the random number generator.

    user_splits : tuple, dict, None or False (default=None)
        If None, it will try the best split given by sklearn Tree.
        If False, this node will not try to split, it will be a leaf node.
        If tuple, it must have the structure:

            >>> user_splits = (feature, split_value)

        If dictionary, it must have the following structure:

            >>> user_splits = {"node": (feature, split_value) or None or False,
            >>>                "left": user_split_lelf,
            >>>                "right": user_split_right}

        At "left" or "right" keys any of the previous options can be set as
        value.

    """
    def __init__(self, split_vars=None, min_weight_fraction_leaf=0.1,
                 test_name="chow", alpha=0.05, max_depth=None, random_state=1,
                 user_splits=None):

        self.split_vars = split_vars
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.test_name = test_name
        self.alpha = alpha
        self.max_depth = max_depth
        self.random_state = random_state
        self.user_splits = user_splits

        self.n_min = None
        self.root = None
        self.metric = None

        if self.max_depth is None:
            self.max_depth = -1

        self._is_fitted = False

    def fit(self, X, y):
        """
        Fits the tree.

        Parameters
        ----------
        X : pandas.DataFrame
            The train dataset.

        y : pandas.DataFrame
            The train target values.
        """

        # check X, y
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas dataframe.")
        if not isinstance(y, pd.DataFrame):
            raise TypeError("y must be a pandas dataframe.")
        check_X_y(X, y.values.ravel())

        # check min_weight_fraction_leaf
        if isinstance(self.min_weight_fraction_leaf, float):
            if not ((self.min_weight_fraction_leaf >= 0.0) and
                    (self.min_weight_fraction_leaf <= 1.0)):
                raise ValueError("min_weight_fraction_leaf must be in "
                                 "range [0,1].")
        else:
            raise TypeError("min_weight_fraction_leaf must be float.")

        # check test_name
        if isinstance(self.test_name, str):
            if not ((self.test_name == "chow") or (self.test_name == "odds")):
                raise ValueError("test_name options are 'odds' or 'chow'.")
        else:
            raise TypeError("test_name must be str.")

        # check alpha
        if isinstance(self.alpha, float):
            if not ((self.alpha > 0.0) and (self.alpha < 1.0)):
                raise ValueError("alpha must be in range (0,1).")
        else:
            raise TypeError("alpha must be float.")

        # check max_depth
        if isinstance(self.max_depth, int):
            if not ((self.max_depth > 0) or (self.max_depth == -1)):
                raise ValueError("max_depth must be a positive integer.")
        else:
            raise TypeError("max_depth must be integer.")

        # check random_state
        if isinstance(self.random_state, (int, np.int32, np.int64)):
            if not (self.random_state >= 0):
                raise ValueError("random_state must be a non negative "
                                 "integer.")
        else:
            raise TypeError("random_state must be integer.")

        self.n_min = int(X.shape[0]*self.min_weight_fraction_leaf)

        self._fit(X, y)

        self._is_fitted = True

    def get_segments(self, X, y=None):
        """
        Gets the segments obtained from the leafs of the fitted tree.

        Parameters
        ----------
        X : pandas.DataFrame
            The train dataset.

        y : pandas.DataFrame
            The train target values.

        Returns
        -------
        segments : list of tuples.
            Each tuple from the list contains: (X_segment, y_segment,
            leaf of the segment)
        """

        if not self._is_fitted:
            raise NotFittedException(self)

        all_leafs = self.root.find_leafs()

        segments = []
        for leaf in all_leafs:
            if y is None:
                segments += [(X[leaf.mask], leaf)]
            else:
                segments += [(X[leaf.mask], y[leaf.mask], leaf)]

        return segments

    def get_all_logs(self):
        return self.root.get_all_logs()

    def _fit(self, X, y):

        # initialize the first mask for the root node.
        root_mask = np.full((X.shape[0]), True)

        # initialize root node.
        self.root = NodeTree(mask=root_mask, parent=None,
                             parent_left_leaf=None,
                             split_vars=self.split_vars,
                             test_name=self.test_name, alpha=self.alpha,
                             random_state=self.random_state,
                             user_splits=self.user_splits)

        # initialize the proposed nodes to be splited.
        nodes_to_split = self.root.get_nodes_to_split()

        # initialize the level of the children nodes within the tree.
        level = 1

        while ((len(nodes_to_split) >= 1) and
               ((level <= self.max_depth) or (self.max_depth == -1))):

            # try to splitt the nodes.
            for node in nodes_to_split:
                node.split_node(X, y, self.n_min)

            # get the mew proposed nodes to be splited.
            nodes_to_split = self.root.get_nodes_to_split()

            # children of the next iteration will be one level deeper.
            level += 1

        # metric of the tree used to evaluate the tree against other trees.
        self._calc_metric(X, y)

    def _calc_metric(self, X, y):
        """Metric to evalueate the tree"""

        # get all the leafs of the tree
        all_leafs = self.root.find_leafs()

        # calculation of the metric
        self.metric = calc_iv(all_leafs, y)
