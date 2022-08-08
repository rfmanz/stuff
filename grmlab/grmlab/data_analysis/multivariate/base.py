"""
Mutlivariate analysis
"""

# Author: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2020.

import matplotlib.pyplot as plt
import numbers
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.manifold import smacof
import time
import warnings

from ...core.base import GRMlabProcess
from ...core.exceptions import NotRunException
from ...modelling.feature_selection import SelectOptimize
from ...reporting.util import reporting_output_format
from .correlation import MultivariateCorrelations

STATUS_VIF_HIGH = "VIF_high"
STATUS_GROUP_VIF_HIGH = "Group_VIF_high"
STATUS_EXCLUDED = "Excluded"
STATUS_NOT_SELECTED = "Not_selected"
STATUS_OK = "ok"

STATUS_OPTIONS = [
    STATUS_EXCLUDED, STATUS_VIF_HIGH, STATUS_NOT_SELECTED,
    STATUS_GROUP_VIF_HIGH, STATUS_OK]

STATUS_REMOVE = [
    STATUS_EXCLUDED]

STATUS_REVIEW = [
    STATUS_VIF_HIGH, STATUS_NOT_SELECTED, STATUS_GROUP_VIF_HIGH]

MSG_METRIC_HIGH = "{} value above threshold {:.3f} > {:.3f}"
MSG_GROUP_METRIC_HIGH = "{} {} value above threshold {:.3f} > {:.3f}"
MSG_EXLUDED = "excluded by variable {}"
MSG_NOT_RECOMMENDED = "not selected in group {}"
MSG_RECOMMENDED = "selected in group {}"


def _multivariate_results(dict_variables, dic_groups, step, format):
    """Return run or transform step results as a dataframe or json"""
    order_cols = ["name", "group_id", "recommended_action", "status",
                  "comment", "vif", "connections",
                  "weight", "excluded", "excluded_by"]

    group_cols = ["name", "recommended_action", "status", "comment", "vif",
                  "max_weight", "n_vars", "n_selected", "pca_explainded"]

    # transform
    order_cols_transform = [
        "name", "group_id", "action", "status", "comment",
        "recommended_action", "user_action", "user_comment", "auto_comment"]

    results_var = pd.DataFrame.from_dict(
            dict_variables,
            columns=["group_id", "recommended_action", "status",
                     "comment", "vif", "connections",
                     "weight", "excluded", "excluded_by"],
            orient="index").reset_index().rename(columns={"index": "name"})

    results_gr = pd.DataFrame.from_dict(
            dic_groups,
            columns=["recommended_action", "status",
                     "comment", "vif", "max_weight", "n_vars",
                     "n_selected", "pca_explainded"],
            orient="index").reset_index().rename(columns={"index": "name"})

    if step == "run":
        special_cols = ["comment"]

        for col in special_cols:
            if col not in results_var.columns:
                results_var[col] = np.nan

        var_results = results_var[order_cols]
        var_report = reporting_output_format(var_results, format)

        for col in special_cols:
            if col not in results_gr.columns:
                results_gr[col] = np.nan

        gr_results = results_gr[group_cols]
        gr_report = reporting_output_format(gr_results, format)

        return var_report, gr_report
    else:
        results_trans = pd.DataFrame.from_dict(
            dict_variables,
            columns=["group_id", "action", "status", "comment",
                     "recommended_action",
                     "user_action", "user_comment", "auto_comment"],
            orient="index").reset_index().rename(columns={"index": "name"})

        trans_results = results_trans[order_cols_transform]
        return reporting_output_format(trans_results, format)


class MultivariateAnalysis(GRMlabProcess):
    """Multivariate analysis of the database.

    Parameters
    ----------
    name : str
        Name given to the class. This is only an identifier set by the
        user.

    target : str or None (default=None)
        The name of the variable flagged as target. If None, selection of
        variables within groups will not be executed.

    date : str or None (default=None)
        The name of the variable flagged as date.

    max_vif : float (default=2.6)
        The maximum allowed value for the variance inflation factor (VIF)
        metric.

    vars_metrics : dict or None (default=None)
        Metrics of the variables to be used in the weighting function.

            >>> vars_metrics = {"var_name1": {"metric_1": value_1,
            >>>                               "metric_2": value_2, ...}
            >>>                 "var_name2": {"metric_1": value_1,
            >>>                               "metric_2": value_2, ...},
            >>>                 ...}

        One can use the dictionary on attribute :code:`._dict_variables` of
        class :mod:`grmlab.data_analysis.Bivariate`, since it has
        the same structure.

    metric_weights : dict or None (default=None)
        Defines the weighting function. The keys are the names of the metrics
        to include in the weighting function and the values the weight of
        each metric on the variable weight calculation. Example:

            >>> metric_weights = {"gini": 0.7, "iv": 0.3}

    correlation_method : str (default="pearson")
        Method to calculate the correlation. The options are: 'pearson',
        'kendall', and 'spearman'.

    equal_mincorr : float (default=0.99)
        Minimum correlation between two variables to consider them equal.

    groups_mincorr : float (default=0.6)
        Minimum correlation to link two variables in the same group.

    links_mincorr : float (default=0.4)
        Minimum correlation to display a link between two variables in the
        reporting.

    alpha : float or None (default=None)
        Constant that multiplies the L1 term in the regression to calculate the
        VIF metric. If None then :code:`alpha=0`, which is equivalent to an
        ordinary least square regression.

    n_max_features : int or None (default=None)
        The maximum number of features to select within each group. If None,
        maximum number of features will be selected within each group, taking
        into account constrains. Check
        :mod:`grmlab.modelling.feature_selection.SelectOptimize` for more
        information.

    max_correlation_target : float (default=0.3)
        The maximum absolute correlation between features and target. Check
        :mod:`grmlab.modelling.feature_selection.SelectOptimize` for more
        information.

    excluded : array-like or None (default=None)
        Name of the features to be excluded in the feature selection method
        within each group. Check
        :mod:`grmlab.modelling.feature_selection.SelectOptimize` for more
        information.

    fixed : array-like or None (default=None)
        Name of the features to be included (if a feasible solution exists) in
        the feature selection method within each group. Check
        :mod:`grmlab.modelling.feature_selection.SelectOptimize` for more
        information.

    verbose : boolean (default=True)
        Controls verbosity of output.
    """
    def __init__(self, name, target=None, date=None, max_vif=2.6,
                 vars_metrics=None,
                 metric_weights=None, correlation_method="pearson",
                 equal_mincorr=0.99, groups_mincorr=0.6, links_mincorr=0.4,
                 alpha=None, n_max_features=None,
                 max_correlation_target=0.3, excluded=None, fixed=None,
                 verbose=True):
        # general parameters
        self.name = name
        self.target = target
        self.date = date
        self.max_vif = max_vif
        self.vars_metrics = vars_metrics
        self.metric_weights = metric_weights
        self.correlation_method = correlation_method
        self.equal_mincorr = equal_mincorr
        self.groups_mincorr = groups_mincorr
        self.links_mincorr = links_mincorr
        self.n_components = 1  # fixed. Change in future releases
        self.alpha = alpha
        self.n_max_features = n_max_features
        self.max_correlation_target = max_correlation_target
        self.excluded = [] if excluded is None else excluded
        self.fixed = [] if fixed is None else fixed
        self.verbose = verbose

        # global results
        self._dict_groups = {}
        self._dict_variables = {}

        # correlation results
        self.mcorr = None
        self.g_equal = None
        self.g_groups = None
        self.df_corr_results = None

        # principal components results
        self._data_pca = None
        self.unique_groups = None
        self.no_grp_vars = None

        # time
        self._time_run = 0
        self._time_run_corr = 0
        self._time_run_pca = 0
        self._time_run_vif = 0
        self._time_run_group = 0
        self._time_transform = 0

        # flags
        self._is_run = False
        self._is_transformed = False

    def group_factor_analysis(self, data, group_vec, n_components=None,
                              max_iter=20):
        """
        Factor analysis to study the underlying structure of the data.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataset with all values in numeric format. It should contain
            the variables (columns) that belong to the specified groups.
        group_vec : array-like
            List of groups names to explore its underlying structure.
        n_components : int (dafault=None)
            Number of factors. If None, the number of factors will be equal to
            length of group_vec.
        max_iter : int (default=20)
            Maximum number of iterations.
        """
        if not self._is_run:
            raise NotRunException(self)

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas dataframe.")

        n_groups = len(group_vec)
        if n_components is None:
            n_components = n_groups
        vars_groups = np.array([])
        len_groups = []
        for i in range(n_groups):
            group_var = list(self._dict_groups[group_vec[i]]["variables"])
            vars_groups = np.append(vars_groups, group_var)
            len_groups.append(len(group_var))
        X = data[vars_groups].values

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            fa = FactorAnalysis(n_components=n_components, max_iter=max_iter)
            fa.fit(X)

        plt.figure(figsize=(20/2, 30/2))
        alpha_d = (1-0.3)/n_components
        tot_vars = 0
        max_val = fa.components_.max()
        min_val = fa.components_.min()
        for i in range(n_components):
            plt.barh(vars_groups, fa.components_[i], alpha=1-alpha_d*i,
                     label="F"+str(i))
        for i in range(n_groups):
            tot_vars += len_groups[i]
            if i < (n_groups-1):
                plt.plot([min_val, max_val],
                         [tot_vars-0.5, tot_vars-0.5], "k-.", alpha=0.5)
        plt.legend()
        plt.show()
        plt.close()

    def group_vars_selection(self, name, selected_variables=None,
                             apply_selection=False, data=None,
                             select_optimize=False):
        """
        Group variables selection.

        It shows a table with data to help in the selection process, and if
        :code:`apply_selection=True`, it applies the changes on actions to
        execute the selection in the :code:`transform` step.

        Parameters
        ----------
        name : str
            Name of the group.
        selected_variables : array-like (default=None)
            List of variables within the group selected by the user. If None,
            the automatic selection of variables is used.
        apply_selection : boolean (default=False)
            If True, variables not selected are set with action remove and
            selected variables with action keep. This action will be executed
            in :code:`.transform` method.
        data : pandas.DataFrame (default=None)
            Dataset with all values in numeric format. It should contain
            the variables (columns) that belong to the specified group.
        select_optimize : boolean (dafault=False)
            If True, a recommendation of viable variables will be executed.
            Pandas DataFrame must be provided in data.
        """
        if not self._is_run:
            raise NotRunException(self)

        if ((select_optimize) and (data is None)):
            raise ValueError("Select Optimize can not be executed, "
                             "data is None")

        if ((select_optimize) and (self.target is None)):
            raise ValueError("Select Optimize can not be executed, unknown "
                             "target")

        if data is not None:
            if not isinstance(data, pd.DataFrame):
                raise TypeError("data must be a pandas dataframe.")

        dict_info = self._dict_groups[name]

        # check variables selected are in group
        if selected_variables is not None:
            for var in selected_variables:
                if var not in dict_info["variables"]:
                    raise KeyError("variable {} not in {}".format(
                        var, name))
        else:
            selected_variables = dict_info["selected_variables"]

        vars_name = dict_info["variables"]

        # execute select optimize to recommend viable varaibles
        recommended_left = []
        if select_optimize:

            n_max_features = len(vars_name)
            fixed = selected_variables
            y = data[self.target].values

            optimize = SelectOptimize(
                feature_names=vars_name, method="classification",
                n_max_features=n_max_features,
                max_correlation=self.groups_mincorr,
                max_correlation_target=self.max_correlation_target,
                excluded=[], fixed=fixed)

            optimize.fit(data[vars_name].values, y)
            recommended_left = vars_name[optimize.support_]

        df_impossible = self.mcorr.df_corr[vars_name].T[vars_name][
            selected_variables] > self.groups_mincorr
        vars_impossible = df_impossible.apply(
            lambda x: sum(x), axis=1).fillna(0).values

        # table generation to be printed.
        vars_lat1 = dict_info["fa"].components_[0]
        vars_lat2 = dict_info["fa"].components_[1]

        vars_weight = []
        vars_selected = []
        for key in vars_name:
            vars_weight.append(self._dict_variables[key]["weight"])
            vars_selected.append("OK" if key in selected_variables else "-")

        sort_info = sorted(zip(vars_name, vars_weight, vars_selected,
                               vars_impossible, vars_lat1, vars_lat2),
                           key=(lambda x: x[1]), reverse=True)
        vars_name = [elt[0] for elt in sort_info]
        vars_weight = [elt[1] for elt in sort_info]
        vars_selected = [elt[2] for elt in sort_info]
        vars_impossible = [elt[3] for elt in sort_info]
        vars_lat1 = [elt[4] for elt in sort_info]
        vars_lat2 = [elt[5] for elt in sort_info]

        report_data = []

        len_max = max([len(elt) for elt in vars_name] + [1])
        for i, nm in enumerate(vars_name):
            report_data.append(nm)
            report_data.append(vars_weight[i])
            report_data.append(vars_selected[i])
            if vars_selected[i] == "OK":
                report_data.append("NO" if vars_impossible[i] > 1 else "-")
            else:
                if vars_impossible[i] == 0:
                    if nm in recommended_left:
                        report_data.append("YES*")
                    else:
                        report_data.append("YES")
                else:
                    report_data.append("-")
            report_data.append(vars_lat1[i])
            report_data.append(vars_lat2[i])

        report = (
                  "\033[94m================================================================================\033[0m\n"
                  "\033[1m\033[94m                         GRMlab Multivariate Analysis                       \033[0m\n"
                  "\033[94m================================================================================\033[0m\n"
                  "\n"
                  " \033[1m{}\n"
                  "   Names" + " " * (len_max - 5) + "| weight | selct | viable | lat-1  | lat-2\n"
                  "   --------------------------------------------------------------------------\n" +
                  "".join(["   {:<" + str(len_max) + "}|  {:4.3f} |   {:<2}  |  {:<4}  | {:+4.3f} | {:+4.3f}\n"
                           for elt in range(len(vars_name))]) +
                  "   \033[94m--------------------------------------------------------------------------\033[0m\n"
                  "   * recommended variables"
                  ).format(name, *report_data)

        print(report)

        if apply_selection:
            self._dict_groups[name]["selected_variables"] = selected_variables
            self._dict_groups[name]["n_selected"] = len(selected_variables)
            self._decision_group_engine(name)

    def plot_group_analysis(self, name):
        """
        Return the PCA weights for the variables within the group, and the
        recommended variables to move forward in the modeling process.

        Parameters
        ----------
        name : str
            Group name.
        """
        if not self._is_run:
            raise NotRunException(self)

        print("Group variance contained in PCA: {:.2%}".format(
            100*self._dict_groups[name]["pca"].explained_variance_ratio_.sum())
        )

        var_weight = np.power(self._dict_groups[name]["pca"].components_[0], 2)
        var_reco = [var_weight[i] if elt in
                    self._dict_groups[name]["selected_variables"] else 0
                    for i, elt in enumerate(
                        self._dict_groups[name]["variables"])]

        plt.figure(
            figsize=(20/2, 1+len(self._dict_groups[name]["variables"])/4))

        plt.barh(
            self._dict_groups[name]["variables"], var_weight,
            color="#1464A5")
        plt.barh(
            self._dict_groups[name]["variables"], var_reco,
            color="#2DCCCD", alpha=0.7, label="selected variables")

        plt.ylim([-1, len(self._dict_groups[name]["variables"])])
        plt.title("PCA components weights for " + name)
        plt.legend()
        plt.xlabel("weight on component")
        plt.show()
        plt.close()

    def plot_var_reg(self, name, plot_var="var_explained"):
        """
        Return information of the regression features used to calculate
        variable's VIF.

        Parameters
        ----------
        name : str
            Name of the variable or group. Only variables without group
            have information of the regression. For variables belonging to a
            group use the group name instead.

        plot_var : str (default="var_explained")
            Regression metric to be plotted for each independent variable in
            the regression. Options are: "var_explained" (% of dependent
            variable variance explained by the independent variable),
            "importance" (importance of the independent variable in the
            regression) and "coef" (coefficient of the variable in the
            regression).
        """
        if not self._is_run:
            raise NotRunException(self)

        if name in list(self.unique_groups):
            dict_info = self._dict_groups[name]
        elif name in list(self.no_grp_vars):
            dict_info = self._dict_variables[name]
        else:
            raise ValueError("variable/group name error")

        print("VIF:", dict_info["vif"])

        names_coef = []
        values_coef = []
        for key in dict_info[plot_var]:
            if dict_info["coef"][key] == 0:
                continue
            else:
                names_coef.append(key)
                values_coef.append(dict_info[plot_var][key])

        coef_sorted = sorted(zip(names_coef, values_coef),
                             key=(lambda x: np.abs(x[1])), reverse=False)

        plt.figure(figsize=(20/2, len(values_coef)/4))
        colors = ["#1464A5", "#2DCCCD"]
        plt.barh([elt[0] for elt in coef_sorted],
                 [max(elt[1], 0.0) for elt in coef_sorted],
                 color=colors[1], label="positive " + plot_var)
        plt.barh([elt[0] for elt in coef_sorted],
                 [np.abs(min(elt[1], 0.0)) for elt in coef_sorted],
                 color=colors[0], label="negative " + plot_var)
        plt.ylim([-1, len(values_coef)])
        plt.title("Linear Regression " + plot_var + "s for " + name)
        plt.legend()
        plt.show()
        plt.close()

    def results(self, step="run", format="dataframe"):
        """
        Return information and flags for all variables in the multivariate
        analysis.

        Parameters
        ----------
        step : str or None (default="run")
            Step name, options are "run" and "transform".

        format : str, "dataframe" or "json" (default="dataframe")
            If "dataframe" return pandas.DataFrame. Otherwise, return
            serialized json.
        """
        return _multivariate_results(self._dict_variables, self._dict_groups,
                                     step, format)

    def run(self, data):
        """
        Run multivariate analysis.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataset with all values in numeric format. Strings and missing
            values should be treated previously.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas dataframe.")

        # check whether given target and date are in dataframe
        if self.target is not None and self.target not in list(data.columns):
            raise KeyError("target variable {} not available in dataframe."
                           .format(self.target))

        if self.date is not None and self.date not in list(data.columns):
            raise KeyError("date variable {} not available in dataframe."
                           .format(self.date))

        if not isinstance(self.max_vif, (int, float)):
            raise TypeError("max_vif should be numeric")

        if self.metric_weights is not None:
            for metric in self.metric_weights:
                for var in list(data.columns):
                    if var in [self.target, self.date]:
                        continue
                    if var not in self.vars_metrics:
                        raise KeyError("variable '{}' not included in "
                                       "vars_metrics".format(var))
                    if metric not in self.vars_metrics[var]:
                        raise KeyError(
                            "Metric '{}' not given for variable '{}'".format(
                                metric, var))

        if self.correlation_method not in ("pearson", "kendall", "spearman"):
            raise ValueError("correlation_method option not supported.")

        # check equal_mincorr, groups_mincorr, links_mincorr
        if not (self.equal_mincorr >= 0 and self.equal_mincorr <= 1):
            raise ValueError("equal_mincorr must be in range (0,1)")
        if not (self.groups_mincorr >= 0 and self.groups_mincorr <= 1):
            raise ValueError("groups_mincorr must be in range (0,1)")
        if not (self.links_mincorr >= 0 and self.links_mincorr <= 1):
            raise ValueError("links_mincorr must be in range (0,1)")

        if self.alpha is not None:
            if not isinstance(self.alpha, (int, float)):
                raise TypeError("alpha should be numeric")

        if self.n_max_features is not None and self.n_max_features < 0:
            raise ValueError("n_max_features must be a positive integer.")

        if not isinstance(self.max_correlation_target, numbers.Number) or (
                self.max_correlation_target < 0.0 or
                self.max_correlation_target > 1.0):
            raise ValueError("max_correlation_target must be a positive "
                             "number in [0, 1].".format(
                self.max_correlation_target))

        self._run(data)

        self._is_run = True

    def stat_var(self, name):
        """
        Feature analysis of the regression used to calculate variable's VIF.

        Parameters
        ----------
        name : str
            Name of the variable or group. Only variables without group
            have information of the regression. For variables belonging to a
            group use the group name instead.
        """
        if not self._is_run:
            raise NotRunException(self)

        if name in list(self.unique_groups):
            dict_info = self._dict_groups[name]
        else:
            dict_info = self._dict_variables[name]

        vars_name = []
        vars_var_exp = []
        vars_coef = []
        variable_vif = dict_info["vif"]
        for key in dict_info["coef"]:
            if dict_info["coef"][key] == 0:
                continue
            else:
                vars_name.append(key)
                vars_coef.append(dict_info["coef"][key])
                vars_var_exp.append(dict_info["var_explained"][key])

        sort_info = sorted(zip(vars_name, vars_var_exp, vars_coef),
                           key=(lambda x: x[1]), reverse=True)
        vars_name = [elt[0] for elt in sort_info]
        vars_var_exp = [elt[1] for elt in sort_info]
        vars_coef = [elt[2] for elt in sort_info]

        vars_vif = []
        vars_corr = []
        for nm in vars_name:
            vars_corr.append(self._pca_corr[name][nm])
            if nm in list(self.unique_groups):
                vars_vif.append(self._dict_groups[nm]["vif"])
            else:
                vars_vif.append(self._dict_variables[nm]["vif"])

        report_data = []

        len_max = max([len(elt) for elt in vars_name] + [1])
        val_max = max([elt for elt in vars_var_exp] + [0.01])
        stars_num = int((74 - len_max - 6 - 8 - 7 - 7) / (val_max))
        for i, nm in enumerate(vars_name):
            report_data.append(nm)
            report_data.append(100*vars_corr[i])
            report_data.append(vars_vif[i])
            report_data.append(vars_coef[i])
            report_data.append(100*vars_var_exp[i])
            report_data.append(
                int(round(vars_var_exp[i]*stars_num, 0)) * "*")

        report = (
                  "\033[94m================================================================================\033[0m\n"
                  "\033[1m\033[94m                         GRMlab Multivariate Analysis                       \033[0m\n"
                  "\033[94m================================================================================\033[0m\n"
                  "\n"
                  " \033[1m{} VIF: {:3.2f}\033[0m\n"
                  "   Names" + " " * (len_max - 5) + "| corr | VIF | coef | Variance Explained\n"
                  "   --------------------------------------------------------------------------\n" +
                  "".join(["   {:<" + str(len_max) + "}|{:+5.1f}%| {:3.2f}| {:+3.2f}| {:4.1f}% {}\n"
                           for elt in range(len(vars_var_exp))]) +
                  "   \033[94m--------------------------------------------------------------------------\033[0m\n"
                  ).format(name, variable_vif, *report_data)

        print(report)

    def transform(self, data, mode="basic"):
        """
        Transform input dataset in-place.

        Reduce the raw dataset by removing variables with action flag equal to
        remove.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw dataset.

        mode : str
            Transformation mode, options are "aggressive" and "basic". If
            ``mode=="aggressive"`` variables tagged with action "remove" and
            "review" are dropped. If ``mode=="basic"`` only variables tagged as
            "remove" are dropped.
        """
        if not self._is_run:
            raise NotRunException(self)

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas dataframe.")

        if mode not in ("aggressive", "basic"):
            raise ValueError("mode {} is not supported.".format(mode))

        self._transform_mode = mode

        # remove variables
        time_init = time.perf_counter()
        self._transform_remove(data, mode=mode)

        self._is_transformed = True
        self._time_transform = time.perf_counter() - time_init

        return self

    def _data_principal_components(self, data):

        df_pca = pd.DataFrame(index=data.index).reset_index(drop=True)

        for grp in self.unique_groups:
            if grp == "no_group":
                continue
            self._dict_groups[grp] = {}
            df_pca[grp] = self._pca(data, grp)[0]

        # add no group variables
        self.no_grp_vars = self.df_corr_results[
            self.df_corr_results["group_id"] == "no_group"]["name"]
        df_pca = df_pca.join(data[self.no_grp_vars].reset_index(drop=True))

        return df_pca

    def _decision_group_engine(self, name):
        print("\n** Variables not selected will be set with "
              "action remove **\n")

        vars_name = self._dict_groups[name]["variables"]
        selected_variables = self._dict_groups[name]["selected_variables"]

        for var in vars_name:
            if var in selected_variables:
                msg = MSG_NOT_RECOMMENDED.format(name)
                new_msg = MSG_RECOMMENDED.format(name)
                status = STATUS_OK
                action = "keep"
            else:
                new_msg = MSG_NOT_RECOMMENDED.format(name)
                msg = MSG_RECOMMENDED.format(name)
                status = STATUS_NOT_SELECTED
                action = "remove"

            # change recomendation. Group variable selections are not
            # considered that the user is going against the statistics.
            self._dict_variables[var]["recommended_action"] = action

            # to trace user action
            self.set_variable_action(var, action, comment=new_msg)

            if msg in self._dict_variables[var]["comment"]:
                idx = self._dict_variables[var]["comment"].index(msg)
                self._dict_variables[var]["comment"][idx] = new_msg
                self._dict_variables[var]["status"][idx] = status
            elif new_msg in self._dict_variables[var]["comment"]:
                pass
            else:
                self._dict_variables[var]["comment"].append(new_msg)
                self._dict_variables[var]["status"].append(status)

    def _decision_engine(self):
        """Set action to be taken for each variable."""
        for gr in self.unique_groups:
            if gr == "no_group":
                continue
            # initialize statuses and comments
            self._dict_groups[gr]["status"] = []
            self._dict_groups[gr]["comment"] = []

            # Group VIF flag
            if self._dict_groups[gr]["vif"] > self.max_vif:
                self._dict_groups[gr]["status"].append(
                    STATUS_GROUP_VIF_HIGH)
                self._dict_groups[gr]["comment"].append(
                    MSG_GROUP_METRIC_HIGH.format(
                        gr, "VIF", self._dict_groups[gr]["vif"],
                        self.max_vif))

            status = self._dict_groups[gr]["status"]
            if any(st in STATUS_REMOVE for st in status):
                self._dict_groups[gr]["action"] = "remove"
            elif any(st in STATUS_REVIEW for st in status):
                self._dict_groups[gr]["action"] = "review"
            else:
                self._dict_groups[gr]["action"] = "keep"

            action = self._dict_groups[gr]["action"]
            self._dict_groups[gr]["recommended_action"] = action

        for var in self.mcorr._column_names:
            # initialize statuses and comments
            self._dict_variables[var]["status"] = []
            self._dict_variables[var]["comment"] = []
            self._dict_variables[var]["user_action"] = None
            self._dict_variables[var]["user_comment"] = None
            self._dict_variables[var]["auto_comment"] = None

            # excluded flag
            if self._dict_variables[var]["excluded"]:
                self._dict_variables[var]["status"].append(STATUS_EXCLUDED)
                self._dict_variables[var]["comment"].append(
                    MSG_EXLUDED.format(
                        self._dict_variables[var]["excluded_by"]))

            # VIF flag
            if self._dict_variables[var]["vif"] > self.max_vif:
                self._dict_variables[var]["status"].append(STATUS_VIF_HIGH)
                self._dict_variables[var]["comment"].append(
                    MSG_METRIC_HIGH.format(
                        "VIF", self._dict_variables[var]["vif"], self.max_vif))

            # Group VIF flag
            if self._dict_variables[var]["group_id"] != "no_group":
                group_id = self._dict_variables[var]["group_id"]
                if self._dict_groups[group_id]["vif"] > self.max_vif:
                    self._dict_variables[var]["status"].append(
                        STATUS_GROUP_VIF_HIGH)
                    self._dict_variables[var]["comment"].append(
                        MSG_GROUP_METRIC_HIGH.format(
                            group_id, "VIF",
                            self._dict_groups[group_id]["vif"], self.max_vif))

            # Not selected flag
            if self._dict_variables[var]["selected"] is not None:
                if not self._dict_variables[var]["selected"]:
                    self._dict_variables[var]["status"].append(
                        STATUS_NOT_SELECTED)
                    self._dict_variables[var]["comment"].append(
                        MSG_NOT_RECOMMENDED.format(
                            self._dict_variables[var]["group_id"]))

            status = self._dict_variables[var]["status"]
            if any(st in STATUS_REMOVE for st in status):
                self._dict_variables[var]["action"] = "remove"
            elif any(st in STATUS_REVIEW for st in status):
                self._dict_variables[var]["action"] = "review"
            else:
                self._dict_variables[var]["action"] = "keep"

            action = self._dict_variables[var]["action"]
            self._dict_variables[var]["recommended_action"] = action

    def _global_vif(self, data):

        X = self._data_pca.values
        std_features = np.std(X, axis=0)
        variables_names = self._data_pca.columns.values

        flag_regression = True
        if len(variables_names) < 2:
            flag_regression = False

        for i, var in enumerate(variables_names):
            if flag_regression:
                y_vif = X.T[i]
                x_vif = np.delete(X.T, i, 0).T
                predictors_vars = np.delete(variables_names, i)
                if self.alpha is None:
                    reg = LinearRegression().fit(x_vif, y_vif)
                else:
                    reg = Lasso(alpha=self.alpha).fit(x_vif, y_vif)

                coef = list(reg.coef_)
                weights = np.abs(coef) * np.delete(std_features, i)
                score = reg.score(x_vif, y_vif)

                if (weights.sum() == 0.0):
                    norm_weights = weights
                else:
                    norm_weights = weights / weights.sum()
                var_explained = norm_weights*reg.score(x_vif, y_vif)
            else:
                score = 0
                predictors_vars = []
                coef = []
                norm_weights = []
                var_explained = []


            if var in list(self.unique_groups):
                self._dict_groups[var]["vif"] = 1/(1-score)
                self._dict_groups[var]["coef"] = dict(
                    zip(predictors_vars, coef))
                self._dict_groups[var]["importance"] = dict(
                    zip(predictors_vars, norm_weights))
                self._dict_groups[var]["var_explained"] = dict(
                    zip(predictors_vars, var_explained))
            elif var in list(self.no_grp_vars):
                self._dict_variables[var]["vif"] = 1/(1-score)
                self._dict_variables[var]["coef"] = dict(
                    zip(predictors_vars, coef))
                self._dict_variables[var]["importance"] = dict(
                    zip(predictors_vars, norm_weights))
                self._dict_variables[var]["var_explained"] = dict(
                    zip(predictors_vars, var_explained))
            else:
                continue

    def _group_analysis(self, data):

        # Canonical Correlation Analysis
        self._group_var_recommender(data)

        # Group multidimensional scaling (mds)
        self._group_mds()

    def _group_mds(self):

        for gr in self.unique_groups:

            if gr == "no_group":
                continue

            vars_in_group = self._dict_groups[gr]["variables"]
            gr_distances = self.mcorr.df_euclidean_distance[
                vars_in_group].T[vars_in_group].T

            mds = dict(zip(vars_in_group,
                           smacof(gr_distances, metric=True)[0]))

            self._dict_groups[gr]["mds"] = mds

    def _group_var_recommender(self, data):

        if self.target is not None:
            y = data[self.target].values

            for gr in self.unique_groups:

                if gr == "no_group":
                    continue

                gr_vars = self._dict_groups[gr]["variables"]

                if self.n_max_features is None:
                    n_max_features = len(gr_vars)
                else:
                    n_max_features = min(len(gr_vars), self.n_max_features)

                excluded = list(set(gr_vars).intersection(self.excluded))
                fixed = list(set(gr_vars).intersection(self.fixed))

                optimize = SelectOptimize(
                    feature_names=gr_vars, method="classification",
                    n_max_features=n_max_features,
                    max_correlation=self.groups_mincorr,
                    max_correlation_target=self.max_correlation_target,
                    excluded=excluded, fixed=fixed)

                optimize.fit(data[gr_vars].values, y)
                self._dict_groups[gr][
                    "selected_variables"] = gr_vars[optimize.support_]
                self._dict_groups[gr]["n_selected"] = self._dict_groups[gr][
                    "selected_variables"].shape[0]

                for var in gr_vars:
                    if var in self._dict_groups[gr]["selected_variables"]:
                        self._dict_variables[var]["selected"] = True
                    else:
                        self._dict_variables[var]["selected"] = False

        else:
            for gr in self.unique_groups:

                if gr == "no_group":
                    continue

                gr_vars = self._dict_groups[gr]["variables"]

                excluded = list(set(gr_vars).intersection(self.excluded))
                fixed = list(set(gr_vars).intersection(self.fixed))

                self._dict_groups[gr][
                    "selected_variables"] = fixed
                self._dict_groups[gr]["n_selected"] = len(fixed)

                for var in gr_vars:
                    if var in self._dict_groups[gr]["selected_variables"]:
                        self._dict_variables[var]["selected"] = True
                    else:
                        self._dict_variables[var]["selected"] = False

    def _pca(self, data, group):

        grp_vars = self.df_corr_results[
            self.df_corr_results["group_id"] == group]["name"]

        n_components = min(len(grp_vars), self.n_components)

        pca = PCA(n_components=n_components)
        pca.fit(data[grp_vars])

        # components and max_iter fixed. These are only for informational
        # purposes.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            fa = FactorAnalysis(n_components=2, max_iter=20)
            fa.fit(data[grp_vars])

        self._dict_groups[group]["pca"] = pca
        self._dict_groups[group]["fa"] = fa
        self._dict_groups[group]["pca_explainded"] = self._dict_groups[group][
            "pca"].explained_variance_ratio_.sum()
        self._dict_groups[group]["variables"] = grp_vars.values
        self._dict_groups[group]["n_vars"] = grp_vars.values.shape[0]
        self._dict_groups[group]["max_weight"] = max(
            self._dict_variables[var]["weight"] for var in grp_vars.values)

        return pca.transform(data[grp_vars]).T

    def _run(self, data):

        time_init = time.perf_counter()

        print("running correlation analysis....")
        self.mcorr = MultivariateCorrelations(
            name=self.name, vars_metrics=self.vars_metrics,
            metric_weights=self.metric_weights,
            correlation_method=self.correlation_method,
            equal_mincorr=self.equal_mincorr,
            groups_mincorr=self.groups_mincorr,
            links_mincorr=self.links_mincorr, verbose=self.verbose)

        columns_drop = [elt for elt in [self.target, self.date]
                        if elt is not None]
        self.g_equal, self.g_groups = self.mcorr.run(
            data.drop(columns=columns_drop))

        self._time_run_corr = time.perf_counter() - time_init

        # use _dict_variables from MultivariateCorrelations (not copy)
        self._dict_variables = self.mcorr._dict_variables
        for key in self._dict_variables:
            self._dict_variables[key]["vif"] = 0.0
            self._dict_variables[key]["selected"] = None
            self._dict_variables[key]["gini"] = 0.0
            self._dict_variables[key]["coef"] = {}
            self._dict_variables[key]["importance"] = {}
            self._dict_variables[key]["var_explained"] = {}

        self.df_corr_results = self.mcorr.results()
        # remove exculded vars
        self.df_corr_results = self.df_corr_results[
            self.df_corr_results["excluded"] == False]

        self.unique_groups = np.unique(self.df_corr_results["group_id"])

        print("obtaining principal components....")
        self._data_pca = self._data_principal_components(data)
        self._pca_corr = self._data_pca.corr()

        self._time_run_pca = (time.perf_counter() - time_init -
                              self._time_run_corr)

        print("global variable/group vif calculation....")
        self._global_vif(data)

        self._time_run_vif = (time.perf_counter() - time_init -
                              self._time_run_corr - self._time_run_pca)

        print("running group analysis....")
        self._group_analysis(data)

        self._time_run_group = (
            time.perf_counter() - time_init - self._time_run_corr -
            self._time_run_pca - self._time_run_vif)

        self._decision_engine()

        self._time_run = time.perf_counter() - time_init

    def _transform_remove(self, data, mode):
        """Remove unnecessary variables in-place."""
        AGGR_COMMENT = "auto: remove due to aggressive mode"
        BASIC_COMMENT = "auto: keep due to basic mode"
        DATE_COMMENT = "auto: keep date column"
        TARGET_COMMENT = "auto: keep target column"

        remove = [var for var in self.mcorr._column_names if
                  self._dict_variables[var]["action"] == "remove"]

        review = [var for var in self.mcorr._column_names if
                  self._dict_variables[var]["action"] == "review"]

        if mode == "aggressive":
            if self.verbose:
                print("columns with action='remove' or action='review' "
                      "except the date and target column if supplied are "
                      "dropped.")

            for var in review:
                if self.date == var:
                    self._dict_variables[var]["action"] = "keep"
                    self._dict_variables[var]["auto_comment"] = DATE_COMMENT
                elif self.target == var:
                    self._dict_variables[var]["action"] = "keep"
                    msg = TARGET_COMMENT
                    self._dict_variables[var]["auto_comment"] = msg
                else:
                    remove.append(var)
                    self._dict_variables[var]["action"] = "remove"
                    self._dict_variables[var]["auto_comment"] = AGGR_COMMENT

        elif mode == "basic":
            if self.verbose:
                print("only columns with action='remove' are dropped.")

            # if recommended action is not removed then keep
            for var in review:
                if self.date == var:
                    msg = DATE_COMMENT
                    self._dict_variables[var]["auto_comment"] = msg
                elif self.target == var:
                    msg = TARGET_COMMENT
                    self._dict_variables[var]["auto_comment"] = msg
                else:
                    msg = BASIC_COMMENT
                    self._dict_variables[var]["auto_comment"] = msg

                self._dict_variables[var]["action"] = "keep"

        self._n_vars_remove = len(remove)

        # drop column in-place
        data.drop(remove, axis=1, inplace=True)

    def get_variable_status(self, variable):
        """
        Get status of the variable.

        Parameters
        ----------
        variable: str
            The variable name.

        Returns
        -------
        status : str
            The status assigned to the variable.
        """
        if not self._is_run:
            raise NotRunException(self)

        if variable not in self._dict_variables.keys():
            raise ValueError("variable {} not in data.".format(variable))
        else:
            return self._dict_variables[variable]["status"]

    def set_variable_status(self, variable, status):
        """
        Set status of the variable.

        Parameters
        ----------
        variable: str
            The variable name.

        status: str
            The status to be set.
        """
        if not self._is_run:
            raise NotRunException(self)

        if variable not in self._dict_variables.keys():
            raise ValueError("variable {} not in data.".format(variable))

        if status not in STATUS_OPTIONS:
            raise ValueError("status {} not supported.".format(status))

        self._dict_variables[variable]["status"] = status

    def get_variable_action(self, variable):
        """
        Get action applied to the variable.

        Parameters
        ----------
        variable: str
            The variable name.

        Returns
        -------
        action : str
            The action assigned to the variable.
        """
        if not self._is_run:
            raise NotRunException(self)

        if variable not in self._dict_variables.keys():
            raise ValueError("variable {} not in data.".format(variable))
        else:
            return self._dict_variables[variable]["action"]

    def set_variable_action(self, variable, action, comment=""):
        """
        Set action applied to the variable.

        Parameters
        ----------
        variable: str
            The variable name.

        action: str
            The action to be set.
        """
        if not self._is_run:
            raise NotRunException(self)

        if variable not in self._dict_variables.keys():
            raise ValueError("variable {} not in data.".format(variable))

        if action not in ("keep", "remove"):
            raise ValueError("action {} not supported.".format(action))

        self._dict_variables[variable]["user_action"] = action
        self._dict_variables[variable]["user_comment"] = comment

        self._dict_variables[variable]["action"] = action
