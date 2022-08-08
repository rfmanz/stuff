"""
Segmentation analysis
"""

# Authors: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2020.

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from packaging import version
from sklearn.metrics import roc_curve, auc
from sklearn import __version__ as sklearn_version
if version.parse(sklearn_version) >= version.parse("0.20.0"):
    from sklearn.impute import SimpleImputer as Imputer
else:
    from sklearn.preprocessing import Imputer

from ...core.base import GRMlabProcess
from ...core.exceptions import NotRunException
from .tree import get_segments
from .util import snappy_model, allison_test


class Segmentation(GRMlabProcess):
    """
    Segmentation analysis of the database.

    Parameters
    ----------
    target : str or None (default=None)
        The name of the variable flagged as target.

    numeric_variables : list or None (default=None)
        List of numeric variables to be considered in the segmentation model.

    categorical_variables : list or None (default=None)
        List of categorical variables to be considered in the segmentation
        model.

    variable_set: list or None (default=None)
        List of variables to use in segmentation. If None, all the
        variables listed in numeric_variables and categorical_variables will be
        used.

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

    nan_strategy : str (default="mean")
        There are three options: "mean", "median", and "most_frequent" to
        replace the missing values.

    max_correlation : float (default=0.6)
        The maximum positive correlation among selected features.

    allison_reg : str (default="logit")
        Type of regression for the Allison test. The two options are: "linear",
        and "logit".

    n_max_features : int or None (default=None)
        Maximum number of features to be selected on quick models. If None,
        the maximum number of features will be selected, taking into account
        constrains. Check
        :mod:`grmlab.modelling.feature_selection.SelectOptimize` for more
        information.

    max_correlation_target : float (default=0.3)
        The maximum absolute correlation between features and target on quick 
        models and Allison test. Check
        :mod:`grmlab.modelling.feature_selection.SelectOptimize` for more
        information.

    excluded : array-like or None (default=None)
        Name of the features to be excluded on all quick models and all Allison
        test. Check
        :mod:`grmlab.modelling.feature_selection.SelectOptimize` for more
        information.

    fixed : array-like or None (default=None)
        Name of the features to be included (if a feasible solution exists) on
        all quick models and all Allison test. Check
        :mod:`grmlab.modelling.feature_selection.SelectOptimize` for more
        information.

    allison_comprehensive : Boolean (default=False)
        If False, Allison test will be executed only between the most similar
        segments. If True, Allison test will be executed for all segment
        combinations, including i-segment with j-segment and j-segment with
        i-segment, since results may differ.

    verbose : int or Boolean (default=False)
        Controls verbosity of output.
    """

    def __init__(self, target, numeric_variables=None,
                 categorical_variables=None, variable_set=None, num_iter=100,
                 min_weight_fraction_leaf=0.1, test_name="chow", alpha=0.05,
                 max_depth=None, random_state=1, user_splits=None,
                 nan_strategy="mean", max_correlation=0.6, allison_reg="logit",
                 n_max_features=None, max_correlation_target=0.3,
                 excluded=None, fixed=None, allison_comprehensive=False,
                 verbose=True):

        self.target = target
        self.numeric_variables = numeric_variables
        self.categorical_variables = categorical_variables
        self.variable_set = variable_set
        self.num_iter = num_iter
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.test_name = test_name
        self.alpha = alpha
        self.max_depth = max_depth
        self.random_state = random_state
        self.user_splits = user_splits
        self.nan_strategy = nan_strategy
        self.max_correlation = max_correlation
        self.allison_reg = allison_reg
        self.n_max_features = n_max_features
        self.max_correlation_target = max_correlation_target
        self.excluded = excluded
        self.fixed = fixed
        self.allison_comprehensive = allison_comprehensive
        self.verbose = verbose

        self.tree = None
        self.vars_in_segmentation = []
        self.results_dict = None
        self.results_allison = None
        self.results_reg_allison = []

        self._is_run = False

        if self.numeric_variables is None:
            self.numeric_variables = []
        if self.categorical_variables is None:
            self.categorical_variables = []

    def plot_results(self):
        """
        Plots the ROC curve for each segment, the combination of all segments,
        and the result if no segments where calculated.
        """

        if self._is_run is False:
            raise NotRunException(self, "run")

        vec_colors = ["#028484", "#5BBEFF", "#666666", "#C49735", "#388D4F"]
        vec_styles = [":", "--", "-."]

        plt.figure(figsize=(15/1, 10/1))

        for i, key in enumerate(self.results_dict):
            if key == "join_segments":
                color = "#C569B9"
                style = "-."
            elif key == "no_segments":
                color = "#1464A5"
                style = "-"
            else:
                color = vec_colors[i % len(vec_colors)]
                style = vec_styles[i % len(vec_styles)]

            plt.plot(self.results_dict[key]["fpr"],
                     self.results_dict[key]["tpr"], style, c=color, label=key)

        plt.plot([0, 1], [0, 1], "-", color="#DA3851", linewidth=1, alpha=0.2)
        plt.title("ROC")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()
        plt.close()

    def get_all_logs(self):
        """
        Return the results of the test in the construction of the segments
        """
        return self.tree.get_all_logs()

    def results(self):
        """
        Return information for each segment.
        """
        if self._is_run is False:
            raise NotRunException(self, "run")

        df_gini = pd.DataFrame(
            {"gini": [self.results_dict[key]["gini"]
             for key in self.results_dict]},
            index=self.results_dict.keys())
        df_gini["increment"] = [
            gin/self.results_dict["no_segments"]["gini"]-1
            for gin in df_gini["gini"]]

        df_vars_coeff = pd.DataFrame(
            columns=["segment","variables", "coef", "importance"])
        for key in self.results_dict:
            if key != "join_segments":
                variables = self.results_dict[key]["variables"].values
                coefs = self.results_dict[key]["model"].coef_[0]
                importance = self.results_dict[key]["feat_importance"]
                df_seg = pd.DataFrame(np.array(
                    [[key]*variables.shape[0], variables,
                     coefs, importance]).T,
                    columns=["segment","variables", "coef", "importance"])
                df_vars_coeff = df_vars_coeff.append(df_seg, ignore_index=True)

        df_allison = pd.DataFrame({
            "group_1": [
                self.results_allison[key]["allison"]["group_1"].shape[0]
                for key in self.results_allison],
            "group_3": [
                self.results_allison[key]["allison"]["group_3"].shape[0]
                for key in self.results_allison],
            "dummy": [self.results_allison[key]["allison"]["dummy"].shape[0]
                      for key in self.results_allison],
            "node_b": [self.results_allison[key]["node_b"]
                       for key in self.results_allison],
            "node_a": [self.results_allison[key]["node_a"]
                       for key in self.results_allison]})[
            ["node_a", "node_b", "dummy", "group_1", "group_3"]]

        return df_gini, df_vars_coeff, df_allison

    def run(self, data):

        # check split_vars
        if isinstance(self.variable_set, list):
            set_vars = set(self.variable_set)
            inter_set = (set_vars & set(data.columns.values))
            if (set_vars != inter_set):
                raise ValueError("Variable names {} are not contained in the "
                                 "dataframe.".format(list(set_vars-inter_set)))
        elif self.variable_set is not None:
            raise TypeError("variable_set must be class list or None, not {}"
                            ".".format(type(self.variable_set)))

        X, y = self._get_X_y(data)

        self._run(X, y)

        self._is_run = True

    def transform(self):
        raise NotImplementedError("Database transformation not implemented.")

    def _compare_result(self, X, y):
        """
        Compare performance among the segmentes. It also compares them with
        the model without segments.
        """

        segments = self.tree.get_segments(X, y)

        mask_all = np.full(X.shape[0], False)

        y_seg = []
        y_pred_seg = []

        self.results_dict = {}

        for seg in segments:

            X_s = X[seg[-1].mask].reset_index(drop=True)
            y_s = y[seg[-1].mask].reset_index(drop=True)

            # description of the segment
            s_name = str(seg[-1].get_description())

            # regression of one segment
            self.results_dict[s_name] = snappy_model(
                X_s, y_s, target=self.target, 
                numeric_variables=self.numeric_variables,
                max_correlation=self.max_correlation,
                n_max_features=self.n_max_features,
                max_correlation_target=self.max_correlation_target,
                excluded=self.excluded, fixed=self.fixed)

            # info to calculate the gini without segments
            mask_all = (mask_all | seg[-1].mask)

            # info to calculate the gini of all segment results
            y_seg = np.concatenate([y_seg, y_s[self.target].values])
            y_pred_seg = np.concatenate(
                [y_pred_seg,
                 [elt[1] for elt in self.results_dict[s_name]["y_pred"]]])

        # all regression
        self.results_dict["no_segments"] = snappy_model(
            X[mask_all], y[mask_all], target=self.target, 
            numeric_variables=self.numeric_variables,
            max_correlation=self.max_correlation,
            n_max_features=self.n_max_features,
            max_correlation_target=self.max_correlation_target,
            excluded=self.excluded, fixed=self.fixed)

        # join segments results
        fpr_seg, tpr_seg, thr = roc_curve(y_seg, y_pred_seg)
        gini_segments = 2*auc(fpr_seg, tpr_seg)-1
        self.results_dict["join_segments"] = {
            "gini": gini_segments, "fpr": fpr_seg, "tpr": tpr_seg,
            "y_pred": y_pred_seg, "variables": None, "model": None}

    def _get_X_y(self, data):
        """
        Define nans and create dummy variables for categorical variables.
        """

        # Nans are filled with the selected strategy only for the numerical
        # variables
        df_imputados = Imputer(missing_values=np.nan,
                               strategy=self.nan_strategy).fit_transform(
            data[self.numeric_variables + [self.target]])
        df_no_nans = pd.DataFrame(
            df_imputados, columns=self.numeric_variables+[self.target])

        df_tree = df_no_nans[self.numeric_variables]

        for num in self.numeric_variables:
            if self.variable_set is None:
                self.vars_in_segmentation += [num]
            elif num in self.variable_set:
                self.vars_in_segmentation += [num]

        for cat in self.categorical_variables:
            X = data[cat].values
            df_dummies = pd.get_dummies(X)
            names = [cat+"_"+elt for elt in df_dummies.columns.values]
            df_dummies.columns = names
            df_tree = df_tree.merge(df_dummies, left_index=True,
                                    right_index=True)
            if self.variable_set is None:
                self.vars_in_segmentation += list(df_dummies.columns.values)
            elif cat in self.variable_set:
                self.vars_in_segmentation += list(df_dummies.columns.values)

        X = df_tree
        y = df_no_nans[[self.target]]

        return X, y

    def _run(self, X, y):

        if self.verbose:
            print("--- Segments derivation")

        self.tree = get_segments(
            X, y, split_vars=self.vars_in_segmentation, num_iter=self.num_iter,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            test_name=self.test_name, alpha=self.alpha,
            random_state=self.random_state, max_depth=self.max_depth,
            user_splits=self.user_splits, verbose=self.verbose)

        if self.verbose:
            print("--- Quick models")

        self._compare_result(X, y)

        if self.verbose:
            print("--- Allison Test")

        self._test_allison(X, y)

    def _test_allison(self, X, y):

        segments_ordered = self.tree.root._order_leafs(y)

        self.results_allison = {}
        if self.allison_comprehensive:
            combinations = itertools.permutations(
                np.arange(0, len(segments_ordered)), 2)
        else:
            combinations = [(i, i+1) for i in range(len(segments_ordered)-1)]

        iter_num = 0
        for (i,j) in combinations:
                self.results_allison[iter_num] = {
                    "node_a": segments_ordered[i][1].get_description(),
                    "node_b": segments_ordered[j][1].get_description()}
                mask1 = segments_ordered[i][1].mask
                mask2 = segments_ordered[j][1].mask

                result, reg_allison = allison_test(
                    X, y, mask1, mask2, target=self.target,
                    numeric_variables=self.numeric_variables,
                    max_correlation=self.max_correlation,
                    max_correlation_target=self.max_correlation_target,
                    type_reg=self.allison_reg,
                    excluded=self.excluded, fixed=self.fixed, verbose=False)

                self.results_allison[iter_num]["allison"] = result
                self.results_allison[iter_num]["allison"][
                    "regression"] = reg_allison
                iter_num += 1
