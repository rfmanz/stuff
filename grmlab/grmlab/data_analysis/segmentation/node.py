"""
Node of the Tree for segment calculation
"""

# Authors: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2020.

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from ...core.base import GRMlabBase
from .util import chow_test, odds_test
from .util import RANDOM_INT


def node_split(X, y, n_min=1, random_state=1, max_depth=6):
    """DecisionTreeClassifier"""
    tree = DecisionTreeClassifier(
        criterion="entropy", splitter="best", max_depth=max_depth,
        min_samples_split=int(2*n_min), min_samples_leaf=n_min,
        min_weight_fraction_leaf=0.1, max_features="auto",
        max_leaf_nodes=None, random_state=random_state,
        min_impurity_decrease=0., class_weight=None)
    tree = tree.fit(X, y)
    return tree


def test_two_trees(X, y, node1, node2, test_name="chow", alpha=0.05):
    """Test between two nodes of the tree"""
    if test_name == "chow":
        X_1 = X[node1.mask].values
        y_1 = y[node1.mask].values
        X_2 = X[node2.mask].values
        y_2 = y[node2.mask].values
        return chow_test(X_1, y_1, X_2, y_2, alpha)
    elif test_name == "odds":
        y_1 = y[node1.mask].values
        y_2 = y[node2.mask].values
        return odds_test(y_1, y_2, alpha)


def get_combinations(val_leafs_01, val_leafs_02):
    """
    All combinations between two groups of nodes sorted by proximity.

    It is sorted from the closest pair to the fardest pair.
    """
    distance = []
    uncle_vals = np.array([val_leaf[0] for val_leaf in val_leafs_02])
    for (val, leaf) in val_leafs_01:
        dif_vals = np.abs(uncle_vals-val)
        idx = np.argmin(dif_vals)
        distance.append((dif_vals[idx], (leaf, val_leafs_02[idx][1])))
    distance_ordered = sorted(distance, key=(lambda x: x[0]))
    return distance_ordered


class NodeTree(GRMlabBase):
    """
    Node class of the tree

    The node class stores information of its parent and children. It also has
    the functions to create the children and to navigate up and down the node
    family structure.

    Parameters
    ----------
    mask : numpy.array.
        Array with values True or False. It selects the records of the train
        database that are in this node of the tree.

    parent : node class (default=None)
        Parent Node. If the node is the root node, set it to None.

    parent_left_leaf : Boolean or None (default=None)
        If True (False), is the left (right) node of the parent. Use None when
        the node is the root node.

    split_vars : list or None (default=None)
        List of variables to split the node. If None, all the variables in the
        dataframe will be used.

    test_name : str (default="chow")
        The name of the test. There are two options: The chow test ('chow') and
        the odds test ('odds')

    alpha : float (default=0.05)
        Significance level of the test.

    random_state : int (default=1)

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
    def __init__(self, mask, parent=None, parent_left_leaf=None,
                 split_vars=None, test_name="chow", alpha=0.05,
                 random_state=1, user_splits=None):
        self.parent_node = parent
        self.mask = mask
        self.parent_left_leaf = parent_left_leaf
        self.split_vars = split_vars
        self.random_state = random_state
        self.test_name = test_name
        self.alpha = alpha
        self.user_splits = user_splits

        np.random.seed(self.random_state)

        # set level
        if self.parent_node is None:
            self.level = 0
        else:
            self.level = self.parent_node.level + 1

        # set user_splits
        if self.user_splits is None:
            self.user_splits = {"node": None, "left": None, "right": None}
        elif self.user_splits is False:
            self.user_splits = {"node": False, "left": None, "right": None}
        elif isinstance(self.user_splits, tuple):
            self.user_splits = {"node": self.user_splits, "left": None,
                                "right": None}

        self.has_children = None
        self.children_nodes = {"left": None, "right": None}
        self.parent_leaf = None
        self.feature = None
        self.value = None
        self.child_test = None
        self.tree = None
        self.test_log = []
        self.sse = None

    def find_leafs(self):
        """
        Look for and returns the leafs with origin in this node
        """
        if (self.children_nodes["left"] is not None):
            # recursively look for leafs in children nodes
            leafs = self.children_nodes["left"].find_leafs()
            leafs += self.children_nodes["right"].find_leafs()
        else:
            # if it has no children, it is a leaf and returns itself.
            return [self]
        return leafs

    def get_description(self):
        """
        Returns the description of the filters made to get to this node from
        the root node.
        """
        description = []
        # Check if it has parent
        if self.parent_node is not None:
            # description according to left or right side.
            if self.parent_left_leaf:
                description += [(self.parent_node.feature + ": <= " +
                                 str(self.parent_node.value))]
            else:
                description += [(self.parent_node.feature + ": > " +
                                 str(self.parent_node.value))]
            # recursive call to the description of the parent node.
            description += self.parent_node.get_description()
        else:
            return []
        return description

    def get_nodes_to_split(self):
        """
        Look for and returns the leafs with origin in this node which has never
        been tried to split.
        """
        nodes_to_split = []
        # if has_children is None means that it has never been proposed for a
        # split before. (otherwise it will be True or False)
        if self.has_children is None:
            return [self]
        elif self.has_children:
            # recursively look for leaf nodes to split in the children nodes.
            nodes_to_split += self.children_nodes["left"].get_nodes_to_split()
            nodes_to_split += self.children_nodes["right"].get_nodes_to_split()
            return nodes_to_split
        else:
            return []

    def show_log(self):
        """Return the log with the results of the tests."""
        return pd.DataFrame(
            self.test_log,
            columns=["feature", "value", "n_left", "n_right", "mean_left",
                     "mean_right", "test", "p_value", "relation", "result"])

    def get_all_logs(self):
        """Returns its logs and their children logs"""
        df_log = self.show_log()
        df_log["description"] = str(self.get_description())
        df_log["level"] = self.level
        if self.parent_left_leaf is not None:
            df_log["side"] = "left" if self.parent_left_leaf else "right"
        else:
            df_log["side"] = None

        if self.has_children:
            # recursively look for leaf nodes to split in the children nodes.
            df_log_left = self.children_nodes["left"].get_all_logs()
            df_log_right = self.children_nodes["right"].get_all_logs()

            return pd.concat([df_log, df_log_left, df_log_right],
                             ignore_index=True, sort=False)
        else:
            return df_log

    def split_node(self, X, y, n_min):
        """Method to split the node

        Parameters
        ----------
        X : pandas.DataFrame
            The train dataset.

        y : pandas.DataFrame
            The train target values.

        n_min : int
            Minimum number of records per leaf.
        """

        # records within this node.
        X_node = X[self.mask]
        y_node = y[self.mask]
        if self.split_vars is not None:
            X_node = X_node[self.split_vars]

        if self.user_splits["node"] is None:
            self.tree = node_split(
                X_node, y_node,
                n_min=n_min,
                random_state=self.random_state,
                max_depth=1)

            # check if the decission tree has detected any split
            if self.tree.tree_.feature.shape[0] == 3:

                # feature used to split the node
                self.feature = X_node.columns[self.tree.tree_.feature[0]]

                # value of the feature to split
                self.value = self.tree.tree_.threshold[0]

                # create the new masks of the children
                mask_left = (self.mask) & (X[self.feature] <= self.value)
                mask_right = (self.mask) & (X[self.feature] > self.value)

                # children nodes
                self.children_nodes["left"] = NodeTree(
                    mask=mask_left, parent=self, parent_left_leaf=True,
                    split_vars=self.split_vars,
                    test_name=self.test_name, alpha=self.alpha,
                    random_state=np.random.randint(RANDOM_INT),
                    user_splits=self.user_splits["left"])
                self.children_nodes["right"] = NodeTree(
                    mask=mask_right, parent=self, parent_left_leaf=False,
                    split_vars=self.split_vars,
                    test_name=self.test_name, alpha=self.alpha,
                    random_state=np.random.randint(RANDOM_INT),
                    user_splits=self.user_splits["right"])
        elif self.user_splits["node"] is False:
            pass
        else:
            self.feature = self.user_splits["node"][0]
            self.value = self.user_splits["node"][1]

            mask_left = (self.mask) & (X[self.feature] <= self.value)
            mask_right = (self.mask) & (X[self.feature] > self.value)

            if (sum(mask_left) >= n_min) and (sum(mask_right) >= n_min):
                # children nodes
                self.children_nodes["left"] = NodeTree(
                    mask=mask_left, parent=self, parent_left_leaf=True,
                    split_vars=self.split_vars,
                    test_name=self.test_name, alpha=self.alpha,
                    random_state=np.random.randint(RANDOM_INT),
                    user_splits=self.user_splits["left"])
                self.children_nodes["right"] = NodeTree(
                    mask=mask_right, parent=self, parent_left_leaf=False,
                    split_vars=self.split_vars,
                    test_name=self.test_name, alpha=self.alpha,
                    random_state=np.random.randint(RANDOM_INT),
                    user_splits=self.user_splits["right"])
            else:
                self._save_log(
                    feature=self.feature, value=self.value,
                    n_left=sum(mask_left), n_right=sum(mask_right),
                    mean_left=y[mask_left].mean()[0],
                    mean_right=y[mask_right].mean()[0],
                    result="Fail")

        if ((self.children_nodes["left"] is not None) and
                (self.children_nodes["right"] is not None)):
            # test between the children
            self._test_childs(X, y)

            if self.child_test[0]:

                self._save_log(
                    feature=self.feature, value=self.value,
                    n_left=sum(mask_left), n_right=sum(mask_right),
                    mean_left=y[mask_left].mean()[0],
                    mean_right=y[mask_right].mean()[0],
                    test=self.test_name, p_value=self.child_test[1],
                    relation="children", result="Pass")

                # check for uncles and cousins
                if self.parent_node is None:
                    # No more checks are needed
                    self.has_children = True
                else:
                    # get all leafs appart from the children
                    family = self._get_family(y)
                    children_level = self.level + 1

                    # leafs at the same level as children
                    cousins = [elt for elt in family
                               if elt[1].level == children_level]
                    # leafs at a lower level
                    uncles = [elt for elt in family
                              if elt[1].level < children_level]

                    children = self._order_leafs(y)

                    # check if there are uncle leafs
                    if len(uncles) >= 1:
                        # get the most similar pair children-uncle
                        uncle_children = get_combinations(children, uncles)[0]
                        child = uncle_children[1][0]
                        uncle = uncle_children[1][1]
                        # test
                        result_u, p_value = test_two_trees(
                            X, y, child, uncle, self.test_name, self.alpha)
                        if not result_u:
                            # close the node if the test returns False
                            self._close_node()
                            self._save_log(
                                n_left=sum(child.mask),
                                n_right=sum(uncle.mask),
                                mean_left=y[child.mask].mean()[0],
                                mean_right=y[uncle.mask].mean()[0],
                                test=self.test_name,
                                p_value=p_value,
                                relation="uncle", result="Fail")
                        else:
                            self._save_log(
                                n_left=sum(child.mask),
                                n_right=sum(uncle.mask),
                                mean_left=y[child.mask].mean()[0],
                                mean_right=y[uncle.mask].mean()[0],
                                test=self.test_name,
                                p_value=p_value,
                                relation="uncle", result="Pass")
                    else:
                        result_u = True

                    # check if there are cousing leafs
                    if len(cousins) >= 1:
                        # get the most similar pair children-cousing
                        cousin_children = get_combinations(children,
                                                           cousins)[0]
                        child = cousin_children[1][0]
                        cousin = cousin_children[1][1]
                        # test
                        result_c, p_value = test_two_trees(
                            X, y, child, cousin, self.test_name, self.alpha)
                        if not result_c:
                            # if the test returns False, close the node with
                            # the smallest number of records.
                            if sum(self.mask) < sum(cousin.parent_node.mask):
                                self._close_node()
                            else:
                                cousin.parent_node._close_node()
                            self._save_log(
                                n_left=sum(child.mask),
                                n_right=sum(cousin.mask),
                                mean_left=y[child.mask].mean()[0],
                                mean_right=y[cousin.mask].mean()[0],
                                test=self.test_name,
                                p_value=p_value,
                                relation="cousin", result="Fail")
                        else:
                            self._save_log(
                                n_left=sum(child.mask),
                                n_right=sum(cousin.mask),
                                mean_left=y[child.mask].mean()[0],
                                mean_right=y[cousin.mask].mean()[0],
                                test=self.test_name,
                                p_value=p_value,
                                relation="cousin", result="Pass")
                    else:
                        result_c = True

                    if result_c and result_u:
                        self.has_children = True
            else:
                self._close_node()
                self._save_log(
                    feature=self.feature, value=self.value,
                    n_left=sum(mask_left), n_right=sum(mask_right),
                    mean_left=y[mask_left].mean()[0],
                    mean_right=y[mask_right].mean()[0],
                    test=self.test_name, p_value=self.child_test[1],
                    relation="children", result="Fail")
        else:
            self.has_children = False
            self._save_log(relation="children", result="Fail")

    def _close_node(self):
        """Close the node deleting the children"""
        self.has_children = False
        self.children_nodes["right"] = None
        self.children_nodes["left"] = None

    def _get_family(self, y):
        """
        Returns all leafs from the tree that are in different branches.
        """
        family_leaf = []
        if self.parent_node is not None:
            # check for the side of his brother node
            bro_side = "right" if self.parent_left_leaf else "left"
            bro = self.parent_node.children_nodes[bro_side]

            # get the leafs of his brother
            family_leaf += bro._order_leafs(y)

            # recursively get the family leafs of the parent.
            family_leaf += self.parent_node._get_family(y)
        return family_leaf

    def _order_leafs(self, y):
        """
        Return the leafs with origin in this node, sorted by the mean value of
        the target.
        """
        leafs = self.find_leafs()
        leafs_ordered = sorted(
            [(y[leaf.mask].mean()[0], leaf) for leaf in leafs],
            key=(lambda x: x[0]))
        return leafs_ordered

    def _save_log(self, feature=None, value=None, n_left=None, n_right=None,
                  mean_left=None, mean_right=None, test=None, p_value=None,
                  relation=None, result=None):
        self.test_log += [[
            feature, value, n_left, n_right, mean_left, mean_right, test,
            p_value, relation, result]]

    def _test_childs(self, X, y):
        """Test for the children nodes"""

        # check if there are children nodes
        if ((self.children_nodes["left"] is not None) and
                (self.children_nodes["right"] is not None)):
            # Test
            self.child_test = test_two_trees(
                X, y, self.children_nodes["left"],
                self.children_nodes["right"], self.test_name, self.alpha)
