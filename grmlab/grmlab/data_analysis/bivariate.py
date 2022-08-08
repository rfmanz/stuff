"""
Bivariate analysis.

References:
    [1] BBVA - GRM. Norma Específica. Construcción de Modelos de Calificación.
        Riesgo de Crédito. Section 10.4.4. - 538. (2017).
"""

# Authors:
#   Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
#   Carlos Gonzalez Berrendero <c.gonzalez.berrender@bbva.com>
#   Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import gc
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from scipy.stats import ks_2samp
from scipy.stats import median_test
from scipy.stats import wasserstein_distance

from ..core.base import GRMlabProcess
from ..core.dtypes import check_date_format
from ..core.dtypes import check_dtype
from ..core.dtypes import check_target_dtype
from ..core.exceptions import NotRunException
from ..data_processing.feature_binning import OptimalGrouping
from ..data_processing.feature_binning import plot
from ..data_processing.feature_binning import table
from ..reporting.util import reporting_output_format
from .util import js_divergence_multivariate
from .util import kruskalwallis
from .util import weighted_quantile
from .util import weighted_value_counts


if not sys.warnoptions:
    warnings.simplefilter("ignore")


STATUS_IV_LOW = "IV_low"
STATUS_IV_HIGH = "IV_high"
STATUS_GINI_LOW = "Gini_low"
STATUS_GINI_HIGH = "Gini_high"
STATUS_PVALUE_CHI2_HIGH = "pvalue_chi2_high"
STATUS_PVALUE_MAX_CHI2_HIGH = "pvalue_max_chi2_high"
STATUS_CRAMER_LOW = "cramer_v_low"

STATUS_PVALUE_ANOVA_HIGH = "pvalue_anova_high"
STATUS_PVALUE_KS_HIGH = "pvalue_ks_high"
STATUS_PVALUE_KW_HIGH = "pvalue_kw_high"
STATUS_PVALUE_MEDIAN_T_HIGH = "pvalue_median_t_high"
STATUS_UNDEFINED = "undefined"

STATUS_OK = "ok"

STATUS_OPTIONS = [
    STATUS_IV_LOW, STATUS_IV_HIGH, STATUS_GINI_LOW, STATUS_GINI_HIGH,
    STATUS_PVALUE_CHI2_HIGH, STATUS_PVALUE_MAX_CHI2_HIGH, STATUS_CRAMER_LOW,
    STATUS_OK, STATUS_PVALUE_ANOVA_HIGH, STATUS_PVALUE_KS_HIGH,
    STATUS_PVALUE_KW_HIGH, STATUS_PVALUE_MEDIAN_T_HIGH, STATUS_UNDEFINED]

STATUS_REMOVE = [
    STATUS_IV_LOW, STATUS_IV_HIGH, STATUS_GINI_LOW, STATUS_GINI_HIGH,
    STATUS_UNDEFINED]

STATUS_REVIEW = [
    STATUS_PVALUE_ANOVA_HIGH, STATUS_PVALUE_KS_HIGH, STATUS_PVALUE_KW_HIGH,
    STATUS_PVALUE_MEDIAN_T_HIGH, STATUS_PVALUE_CHI2_HIGH,
    STATUS_PVALUE_MAX_CHI2_HIGH, STATUS_CRAMER_LOW]


def _calculate_gini_from_binning_table(binning_table):
    r"""
    Calculate Gini as SAS Enterprise Miner Gini index:

    .. math::

        \text{Gini} = 1-\frac{2\sum_{i=1}^{m}\left(n_i^{\text{event}}
        \sum_{j=0}^{i-1}n_j^{\text{non-event}}\right)+
        \sum_{i=0}^m \left(n_i^{\text{event}}n_i^{\text{non-event}}
        \right)}{N^{\text{event}}N^{\text{non-event}}}

    where :math:`m` is the number of groups. The groups are sorted in
    descending order of the proportion of events.
    """
    event = binning_table["Event count"].values
    nonevent = binning_table["Non-event count"].values

    total_event = event[-1]
    total_nonevent = nonevent[-1]
    event = event[:-1]
    nonevent = nonevent[:-1]

    # filter empty groups and sort groups by descending order of the proportion
    # of events within the group.
    filtered_table = filter(lambda x: (x[0] + x[1]) > 0, zip(event, nonevent))
    sorted_table = sorted(filtered_table, key=lambda x: x[1] / (x[0] + x[1]))
    ev, nev = map(list, zip(*sorted_table))

    n = len(ev)
    sum_1 = sum(ev[i] * nev[i] for i in range(n))
    sum_2 = sum(ev[i] * sum(nev[j] for j in range(i)) for i in range(1, n))

    return 1 - (sum_1 + 2*sum_2) / (total_event * total_nonevent)


def _variable_descriptive_numerical(name, data):
    report = (
        "---------------------------------------------------------------------\n"
        "                         Descriptive analysis                        \n"
        "---------------------------------------------------------------------\n"
        "   variable: {}\n"
        "\n"
        "   Binning information              Statistics\n"
        "     IV                 {:>7.5f}       p-value Chi^2           {:>6.5f}\n"
        "     gini               {:>7.5f}       p-value Chi^2 max       {:>6.5f}\n"
        "     PD monotonicity {:>10}       Cramer's V            {:>.3E}\n"
        "     groups          {:>10}       p-value anova           {:>6.5f}\n"
        "     group special   {:>10}       p-value KS              {:>6.5f}\n"
        "     group missing   {:>10}       p-value kruskal-wallis  {:>6.5f}\n"
        "     group others    {:>10}       p-value median-t        {:>6.5f}\n"
        "                                      divergence            {:>.3E}\n"
        "                                      wasserstein distance  {:>.3E}\n"
        "\n"
        "   Distribution          (    nonevent     |      event   )\n"
        "     mean                   {:>10.3E}       {:>10.3E}\n"
        "     std                    {:>10.3E}       {:>10.3E}\n"
        "     min                    {:>10.3E}       {:>10.3E}\n"
        "     p25                    {:>10.3E}       {:>10.3E}\n"
        "     p50                    {:>10.3E}       {:>10.3E}\n"
        "     p75                    {:>10.3E}       {:>10.3E}\n"
        "     max                    {:>10.3E}       {:>10.3E}\n"
        "---------------------------------------------------------------------\n"
        )

    keys = [
        "iv", "pvalue_chi2", "gini", "pvalue_chi2_max", "pd_monotonicity",
        "cramer_v", "groups", "pvalue_anova", "group_special", "pvalue_ks",
        "group_missing", "pvalue_kw", "group_others", "pvalue_median_t",
        "divergence", "wasserstein_distance", "mean_event", "mean_nonevent",
        "std_event", "std_nonevent", "min_event", "min_nonevent",
        "percentile_25_event", "percentile_25_nonevent", "median_event",
        "median_nonevent", "percentile_75_event", "percentile_75_nonevent",
        "max_event", "max_nonevent"]

    report_data = [name] + list(data[k] for k in keys)
    return report.format(*report_data)


def _variable_descriptive_categorical(name, data):
    report = (
        "---------------------------------------------------------------------\n"
        "                         Descriptive analysis                        \n"
        "---------------------------------------------------------------------\n"
        "   variable: {}\n"
        "\n"
        "   Binning information              Statistics\n"
        "     IV                 {:>7.5f}       p-value Chi^2           {:>6.5f}\n"
        "     gini               {:>7.5f}       p-value Chi^2 max       {:>6.5f}\n"
        "     PD monotonicity {:>10}       Cramer's V            {:>.3E}\n"
        "     groups          {:>10}       number of categories     {:>6}\n"
        "     group special   {:>10}       most freq. category  {:>10}\n"
        "     group missing   {:>10}       % most freq. category   {:>7.2%}\n"
        "     group others    {:>10}\n"
        "\n"
        "   Distribution (number of categories by group)\n"
        "     mean                {:>6.1f}\n"
        "     std                 {:>6.1f}\n"
        "     min                 {:>6}\n"
        "     p25                 {:>6.1f}\n"
        "     p50                 {:>6.1f}\n"
        "     p75                 {:>6.1f}\n"
        "     max                 {:>6}\n"
        "---------------------------------------------------------------------\n"
        )

    keys = [
        "iv", "pvalue_chi2", "gini", "pvalue_chi2_max", "pd_monotonicity",
        "cramer_v", "groups", "n_categories", "group_special",
        "most_freq_category", "group_missing", "p_most_freq_category",
        "group_others", "mean_n_categories", "std_n_categories",
        "min_n_categories", "percentile_25_n_categories",
        "median_n_categories", "percentile_75_n_categories",
        "max_n_categories"]

    report_data = [name] + list(data[k] for k in keys)
    return report.format(*report_data)


def _variable_temporal(name, data):
    report = (
        "---------------------------------------------------------------------\n"
        "                          Temporal analysis                          \n"
        "---------------------------------------------------------------------\n"
        "   variable: {}\n"
        "\n"
        "   group size divergence\n"
        "     uniform        {:>5.3f}               exponential      {:>5.3f}\n"
        "\n"
        "   Distribution (Information Value)\n"
        "     mean                {:>6.3f}\n"
        "     std                 {:>6.3f}\n"
        "     min                 {:>6.3f}\n"
        "     p25                 {:>6.3f}\n"
        "     p50                 {:>6.3f}\n"
        "     p75                 {:>6.3f}\n"
        "     max                 {:>6.3f}\n"
        "---------------------------------------------------------------------\n"
        )

    keys = [
        "t_split_div_uniform", "t_split_div_exponential",
        "t_iv_mean", "t_iv_std", "t_iv_min", "t_iv_percentile_25",
        "t_iv_median", "t_iv_percentile_75", "t_iv_max"]

    report_data = [name] + list(data[k] for k in keys)
    return report.format(*report_data)


def _variable_plot_descriptive(optbin, basic_binning_table, basic_splits,
                               plot_type, plot_bar_type):
    # plot optimal binning
    optbin.plot_binning_table(plot_type, plot_bar_type)

    # plot fixed binned variables
    plot(basic_binning_table, basic_splits, plot_type, plot_bar_type)


def _variable_plot_temporal(name, data):
    split_pd = data["t_split_event_by_date"] / data["t_split_count_by_date"]
    t_split_total_by_date = data["t_split_count_by_date"].sum(axis=0)
    split_size = data["t_split_count_by_date"] / t_split_total_by_date
    n_splits, _ = split_pd.shape

    plt.subplot(2, 1, 1)
    for i in range(n_splits):
        plt.plot(split_pd[i, :], label="split {}".format(i))
    plt.ylabel("Event rate")

    plt.subplot(2, 1, 2)
    for i in range(n_splits):
        plt.plot(split_size[i, :], label="split {}".format(i))
    plt.ylabel("Group size")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
               fancybox=True, shadow=True, ncol=3)
    plt.show()

    plt.boxplot(split_pd.tolist(),
                labels=["split {}".format(i) for i in range(n_splits)])
    plt.ylabel("Event rate")
    plt.show()


def _bivariate_results(dict_variables, step, date, format):
    """Return run or transform step results as a dataframe or json"""

    # common metrics
    order_cols = [
        "name", "dtype", "recommended_action", "status", "comment", "iv",
        "gini", "pd_monotonicity", "groups", "group_special", "group_missing",
        "group_others", "pvalue_chi2", "pvalue_chi2_max", "cramer_v",
        "optbin_params", "optbin_comment", "optbin_default"]

    order_cols_transform = [
        "name", "dtype", "action", "status", "comment",
        "recommended_action", "user_action", "user_comment", "auto_comment"]

    results = pd.DataFrame.from_dict(dict_variables).T
    results.reset_index(level=0, inplace=True)
    results.rename(columns={"index": "name"}, inplace=True)

    if step == "run":
        special_cols = [
            "comment", "optbin_params", "optbin_comment", "optbin_default"]

        for col in special_cols:
            if col not in results.columns:
                results[col] = np.nan

        results = results[order_cols]
        return reporting_output_format(results, format)
    else:
        for col in ["comment", "user_action", "user_comment", "auto_comment"]:
            if col not in results.columns:
                results[col] = np.nan

        results = results[order_cols_transform]
        return reporting_output_format(results, format)


def _bivariate_stats(report_data, step):
    if step == "run":
        report = (
            "\033[94m================================================================================\033[0m\n"
            "\033[1m\033[94m                           GRMlab Bivariate 0.1: Run                            \033[0m\n"
            "\033[94m================================================================================\033[0m\n"
            "\n"
            " \033[1mGeneral information                      Configuration options\033[0m\n"
            "   number of samples   {:>8}             special values                {:>3}\n"
            "   number of variables {:>8}             nominal variables             {:>3}\n"
            "   target variable     {:>8}             special handler policy {:>10}\n"
            "   target type         {:>8}             special WoE policy     {:>10}\n"
            "   date variable       {:>8}             missing WoE policy     {:>10}\n"
            "                                            optimal grouping provided     {:>3}\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            "\n"
            " \033[1mVariables analysis                       Findings analysis (general)\033[0m\n"
            "   numerical           {:>8}             low IV / high IV         {:>3} /{:>3}\n"
            "   ordinal             {:>8}             low gini / high gini     {:>3} /{:>3}\n"
            "   categorical         {:>8}             high p-value Chi^2           {:>4}\n"
            "   nominal             {:>8}             high p-value Chi^2 max       {:>4}\n"
            "                                            low Cramer's V               {:>4}\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            "\n"
            " \033[1mTarget analysis                          Findings analysis (num. / ord.)\033[0m\n"
            "   number of events    {:>8}             high p-value anova           {:>4}\n"
            "   number of nonevents {:>8}             high p-value KS              {:>4}\n"
            "   event rate          {:>8.2%}             high p-value kruskal-wallis  {:>4}\n"
            "                                            high p-value median-t        {:>4}\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            "\n"
            " \033[1mTiming statistics\033[0m\n"
            "   total               {:>6.3f}\n"
            "     optimal grouping  {:>6.3f} ({:>5.1%})\n"
            "       numerical         {:>6.3f} ({:>5.1%})\n"
            "       ordinal           {:>6.3f} ({:>5.1%})\n"
            "       categorical       {:>6.3f} ({:>5.1%})\n"
            "       nominal           {:>6.3f} ({:>5.1%})\n"
            "     metrics           {:>6.3f} ({:>5.1%})\n"
            "       numerical         {:>6.3f} ({:>5.1%})\n"
            "       ordinal           {:>6.3f} ({:>5.1%})\n"
            "       categorical       {:>6.3f} ({:>5.1%})\n"
            "       nominal           {:>6.3f} ({:>5.1%})\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            )
    else:
        report = (
            "\033[94m================================================================================\033[0m\n"
            "\033[1m\033[94m                          GRMlab Bivariate 0.1: Transform                           \033[0m\n"
            "\033[94m================================================================================\033[0m\n"
            "\n"
            " \033[1mResults                                        Timing statistics\033[0m\n"
            "   original data       {:>8}                   total        {:>6.3f}\n"
            "   after bivariate     {:>8}\n"
            "   removed             {:>8}\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            "\n"
            "\033[1mUser actions\033[0m\n"
            "   number of actions   {:>8}\n"
            "   number of comments  {:>8} ({:>5.1%})\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            )

    return report.format(*report_data)


class Bivariate(GRMlabProcess):
    """
    Parameters
    ----------
    target : str or None (default=None)
        The name of the variable flagged as target.

    date : str or None (default=None)
        The name of the variable flagged as date.

    optimalgrouping: object or None (default=None)
        A run instance of OptimalGrouping, performing binning of all variables
        in dataset.

    variables_nominal : list or None (default=None)
        List of ordinal variables to be treated as nominal.

    special_values : list or None (default=None)
        List of special values to be considered.

    min_iv : float (default=0.015)
        The minimum allowed value for the Information Value (IV) metric.

    max_iv : float (default=0.7)
        The maximum allowed value for the Information Value (IV) metric.

    min_gini : float (default=0.025)
        The minimum allowed value for the Gini metric.

    max_gini : float (default=0.85)
        The maximum allowed value for the Gini metric.

    max_pvalue_anova : float (default=0.05)
        The maximum allowed value for the p-value of the Anova test.

    max_pvalue_chi2 : float (default=0.05)
        The maximum allowed value for the p-value of the Chi2 test.

    max_pvalue_ks : float (default=0.05)
        The maximum allowed value for the p-value of the Kolmogorov-Smirnov
        test.

    max_pvalue_kw : float (default=0.05)
        The maximum allowed value for the p-value of the Kruskal-Wallis test.

    max_pvalue_median_t : float (default=0.05)
        The maximum allowed value for the p-value of the Median-t test.

    min_cramer_v : float (default=0.1)
        The minimum allowed value for the measure of association Cramer's V.

    monotonicity_force : boolean (default=False)
        Force ascending or descending monotonicity.

    special_handler_policy : str (default="join")
        Method to handle special values. Options are "join", "separate" and
        "binning". Option "join" creates an extra bucket containing all special
        values. Option "separate" creates an extra bucket for each special
        value. Option "binning" performs feature binning of special values
        using ``grmlab.data_processing.feature_binning.CTree`` in order to
        split special values if these are significantly different.

    special_woe_policy : str (default=empirical)
        Weight-of-Evidence (WoE) value to be assigned to special values
        buckets. Options supported are: "empirical", "worst", "zero". Option
        "empirical" assign the actual WoE value. Option "worst" assigns the WoE
        value corresponding to the bucket with the highest event rate. Finally,
        option "zero" assigns value 0.

    missing_woe_policy : str (default="empirical")
        Weight-of-Evidence (WoE) value to be assigned to missing values bucket.
        Options supported are: "empirical", "worst", "zero". Option "empirical"
        assign the actual WoE value. Option "worst" assigns the WoE value
        corresponding to the bucket with the highest event rate. Finally,
        option "zero" assigns value 0.

    optbin_options : dict or None (default=None):
        Dictionary with options and comments to pass to a particular optbin
        instance.

    verbose : int or boolean (default=False)
        Controls verbosity of output.

    See also
    --------
    grmlab.data_processing.feature_binning.OptimalGrouping
    """
    def __init__(self, target, date=None, optimalgrouping=None,
                 variables_nominal=None, special_values=None, min_iv=0.015,
                 max_iv=0.7, min_gini=0.025, max_gini=0.85,
                 max_pvalue_anova=0.05, max_pvalue_chi2=0.05,
                 max_pvalue_ks=0.05, max_pvalue_kw=0.05,
                 max_pvalue_median_t=0.05, min_cramer_v=0.1,
                 monotonicity_force=False, special_handler_policy="join",
                 special_woe_policy="empirical",
                 missing_woe_policy="empirical", optbin_options=None,
                 verbose=False):

        # main input
        self.target = target
        self.date = date
        self.optimalgrouping = optimalgrouping

        # nominal variables
        if variables_nominal is None:
            self.variables_nominal = []
        else:
            self.variables_nominal = variables_nominal

        # special values options
        self.special_values = [] if special_values is None else special_values
        self.special_handler_policy = special_handler_policy
        self.special_woe_policy = special_woe_policy

        # missing values options
        self.missing_woe_policy = missing_woe_policy

        self.min_iv = min_iv
        self.max_iv = max_iv
        self.min_gini = min_gini
        self.max_gini = max_gini
        self.max_pvalue_anova = max_pvalue_anova
        self.max_pvalue_chi2 = max_pvalue_chi2
        self.max_pvalue_ks = max_pvalue_ks
        self.max_pvalue_kw = max_pvalue_kw
        self.max_pvalue_median_t = max_pvalue_median_t
        self.min_cramer_v = min_cramer_v

        # monotonicity
        self.monotonicity_force = monotonicity_force

        # optbin options
        self.optbin_options = optbin_options

        # others
        self.verbose = verbose

        self._n_samples = None
        self._n_vars = None
        self._n_vars_remove = None

        # target information
        self._target = []
        self._target_dtype = None
        self._n_events = None
        self._n_nonevents = None
        self._event_rate = None

        # dates information
        self._dates = []
        self._unique_dates = []

        # auxiliary information
        self._column_names = []
        self._dict_variables = {}

        # timing statistics
        self._time_run = None
        self._time_run_optimalgrouping = None
        self._time_run_metrics = None
        self._time_run_numerical = 0
        self._time_run_ordinal = 0
        self._time_run_categorical = 0
        self._time_run_nominal = 0
        self._time_transform = None

        # transform parameters
        self._transform_mode = None

        # flags
        self._is_optimalgrouping_provided = False
        self._is_run = False
        self._is_transformed = False

    def results(self, step="run", format="dataframe"):
        """
        Return information and flags for all variables binned using OptBin.

        Parameters
        ----------
        step : str or None (default="run")
            Step name, options are "run" and "transform".

        format : str, "dataframe" or "json" (default="dataframe")
            If "dataframe" return pandas.DataFrame. Otherwise, return
            serialized json.
        """
        if step not in ("run", "transform"):
            raise ValueError("step not found.")

        if step == "run" and not self._is_run:
            raise NotRunException(self, "run")
        elif step == "transform" and not self._is_transformed:
            raise NotRunException(self, "transform")

        date = (self.date is not None)
        return _bivariate_results(self._dict_variables, step, date, format)

    def run(self, data, sample_weight=None):
        """
        Run bivariate analysis.

        Note that temporal analysis is only performed if a date column is
        provided during instantiation.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw dataset.

        sample_weight : array-like, shape = [n_samples] (default=None)
            Individual weights for each sample.

        Returns
        -------
        self : object
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas dataframe.")

        # check sample_weight
        if sample_weight is not None:
            if not isinstance(sample_weight, (list, np.ndarray)):
                raise TypeError("sample_weight must be a list or a numpy"
                                "array.")
            else:
                sample_weight = np.array(sample_weight)
                if sample_weight.shape[0] != data.shape[0]:
                    raise ValueError("sample_weight len and data rows number"
                                     "do not match.")

        # check options
        if not isinstance(self.special_values, (list, np.ndarray)):
            raise TypeError("special values must be a list or a numpy array.")

        if not isinstance(self.variables_nominal, (list, np.ndarray)):
            raise TypeError("variables nominal must be a list or numpy array.")

        # special_woe_policy
        if self.special_woe_policy not in ("empirical", "worst", "zero"):
            raise ValueError("special_woe_policy option not supported.")

        # missing_woe_policy
        if self.missing_woe_policy not in ("empirical", "worst", "zero"):
            raise ValueError("missing_woe_policy option not supported.")

        # optimal_grouping
        if self.optimalgrouping is not None:
            if not isinstance(self.optimalgrouping, OptimalGrouping):
                raise TypeError("optimalgrouping {} must be a class "
                                "of type OptimalGrouping"
                                .format(self.optimalgrouping.__name__))
            else:
                self._is_optimalgrouping_provided = True

        self._n_samples = len(data)
        self._n_vars = len(data.columns)
        self._column_names = list(data.columns)

        # check whether given target and date are in dataframe
        if self.target is not None and self.target not in self._column_names:
            raise ValueError("target variable {} not available in dataframe."
                             .format(self.target))

        if self.date is not None and self.date not in self._column_names:
            raise ValueError("date variable {} not available in dataframe."
                             .format(self.date))

        # keep date data for temporal analysis
        if self.date is not None:
            self._dates = data[self.date].values
            # check date format
            check_date_format(self._dates)

            self._unique_dates = np.unique(self._dates)
            self._column_names.remove(self.date)
        else:
            print("date is not provided. Temporal analysis will not be run.")

        # target information
        self._target = data[self.target].values
        self._target_dtype = check_target_dtype(self._target)
        self._column_names.remove(self.target)

        if self._target_dtype == "binary":
            self._n_events = np.count_nonzero(self._target)
            self._n_nonevents = self._n_samples - self._n_events
            self._event_rate = self._n_events / self._n_samples

        # check optimalgrouping parameters
        if self.optimalgrouping is not None:
            if self.optimalgrouping._target_name != self._target_name:
                raise ValueError("target ({}) and optimalgrouping "
                                 "target ({}) must coincide.".format(
                                    self._target_name,
                                    self.optimalgrouping._target_name))

            if self.optimalgrouping._is_run:
                if self._target_dtype != self.optimalgrouping._target_dtype:
                    raise ValueError("target dtype and optimalgrouping "
                                     "dtype must coincide. {} != {}".format(
                                        self._target_dtype,
                                        self.optimalgrouping._target_dtype))

                if self.optimalgrouping._n_samples != self._n_samples:
                    raise ValueError(
                        "target and optimalgrouping target dimensions must "
                        "coincide. {} != {}"
                        .format(self._n_samples,
                                self.optimalgrouping._n_samples))

                if any(self.optimalgrouping._target != self._target):
                    raise ValueError("target and optimalgrouping target must "
                                     "coincide.")

        # run bivariate
        self._run(data, sample_weight)

        self._is_run = True

        return self

    def stats(self, step="run"):
        """
        Bivariate analysis statistics.

        Parameters
        ----------
        step : str or None (default="run")
            Step name, options are "run" and "transform".
        """
        if step not in ("run", "transform"):
            raise ValueError("step not found.")

        if not self._is_run and step == "run":
            raise NotRunException(self, "run")
        elif not self._is_transformed and step == "transform":
            raise NotRunException(self, "transform")

        dict_values = self._dict_variables.values()

        if step == "run":
            special_flag = "yes" if self.special_values else "no"
            nominal_flag = "yes" if self.variables_nominal else "no"
            date_flag = self.date if self.date is not None else "not set"

            if self._is_optimalgrouping_provided:
                optimal_grouping_flag = "yes"
            else:
                optimal_grouping_flag = "no"

            # timing
            perc_time_numerical = self._time_run_numerical / self._time_run
            perc_time_ordinal = self._time_run_ordinal / self._time_run
            perc_time_categorical = self._time_run_categorical / self._time_run
            perc_time_nominal = self._time_run_nominal / self._time_run

            perc_time_og_numerical = (self.optimalgrouping._time_run_numerical
                                      / self.optimalgrouping._time_run)
            perc_time_og_ordinal = (self.optimalgrouping._time_run_ordinal
                                    / self.optimalgrouping._time_run)
            perc_time_og_categorical = (
                self.optimalgrouping._time_run_categorical
                / self.optimalgrouping._time_run)
            perc_time_og_nominal = (self.optimalgrouping._time_run_nominal
                                    / self.optimalgrouping._time_run)

            perc_time_og = self.optimalgrouping._time_run / self._time_run
            time_metrics = self._time_run - self.optimalgrouping._time_run
            perc_time_metrics = time_metrics / self._time_run

            [n_numerical, n_ordinal, n_categorical, n_nominal] = [
                sum(d["dtype"] == dtype for d in dict_values) for dtype
                in ("numerical", "ordinal", "categorical", "nominal")]

            [iv_low, iv_high, gini_low, gini_high, undefined,
                pvalue_anova_high, pvalue_ks_high, pvalue_kw_high,
                pvalue_median_t_high, pvalue_chi2_high, pvalue_chi2_max_high,
                cramer_v_low] = [
                sum(status in d["status"] for d in dict_values)
                for status in STATUS_REMOVE + STATUS_REVIEW]

            # prepare data
            report_data = [
                self._n_samples, special_flag, self._n_vars,
                nominal_flag, self.target, self.special_handler_policy,
                self._target_dtype, self.special_woe_policy, date_flag,
                self.missing_woe_policy, optimal_grouping_flag,
                n_numerical, iv_low, iv_high, n_ordinal, gini_low, gini_high,
                n_categorical, pvalue_chi2_high, n_nominal,
                pvalue_chi2_max_high, cramer_v_low, self._n_events,
                pvalue_anova_high, self._n_nonevents, pvalue_ks_high,
                self._event_rate, pvalue_kw_high, pvalue_median_t_high,
                self._time_run, self.optimalgrouping._time_run,
                perc_time_og, self.optimalgrouping._time_run_numerical,
                perc_time_og_numerical, self.optimalgrouping._time_run_ordinal,
                perc_time_og_ordinal,
                self.optimalgrouping._time_run_categorical,
                perc_time_og_categorical,
                self.optimalgrouping._time_run_nominal, perc_time_og_nominal,
                time_metrics, perc_time_metrics, self._time_run_numerical,
                perc_time_numerical, self._time_run_ordinal, perc_time_ordinal,
                self._time_run_categorical, perc_time_categorical,
                self._time_run_nominal, perc_time_nominal]
        else:
            n_vars_after = self._n_vars - self._n_vars_remove

            n_user_actions = sum(1 for d in dict_values if "user_action" in d)
            n_user_comment = sum(d["user_comment"] != "" for d in dict_values
                                 if "user_action" in d)

            if n_user_actions:
                perc_user_comment = n_user_comment / n_user_actions
            else:
                perc_user_comment = 0

            report_data = [
                self._n_vars, self._time_transform, n_vars_after,
                self._n_vars_remove, n_user_actions, n_user_comment,
                perc_user_comment]

        print(_bivariate_stats(report_data, step))

    def transform(self, data, mode="basic"):
        """
        Transform input dataset in-place.

        Reduce the raw dataset by removing columns with action flag equal to
        remove.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw dataset.

        mode : str
            Transformation mode, options are "aggresive" and "basic". If
            ``mode=="aggressive"`` variables tagged with action "remove" and
            "review" are dropped. If ``mode=="basic"`` only variables tagged as
            "remove" are dropped.

        Returns
        -------
        self : object
        """
        if not self._is_run:
            raise NotRunException(self, "run")

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
            raise NotRunException(self, "run")

        if variable not in self._column_names:
            raise ValueError("variable '{}' not in data.".format(variable))
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
            raise NotRunException(self, "run")

        if variable not in self._column_names:
            raise ValueError("variable '{}' not in data.".format(variable))

        if status not in STATUS_OPTIONS:
            raise ValueError("status '{}' not supported.".format(status))

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
            raise NotRunException(self, "run")

        if variable not in self._column_names:
            raise ValueError("variable '{}' not in data.".format(variable))
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
            raise NotRunException(self, "run")

        if variable not in self._column_names:
            raise ValueError("variable '{}' not in data.".format(variable))

        if action not in ("keep", "remove"):
            raise ValueError("action '{}' not supported.".format(action))

        self._dict_variables[variable]["user_action"] = action
        self._dict_variables[variable]["user_comment"] = comment

        self._dict_variables[variable]["action"] = action

    def variable_stats_descriptive(self, variable):
        """
        Generate a detailed descriptive analysis report for a given variable.

        Parameters
        ----------
        variable : str
            The variable name.
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        if variable not in self._column_names:
            raise ValueError("variable '{}' not in data.".format(variable))

        data = self._dict_variables[variable]

        if data["dtype"] in ("numerical", "ordinal"):
            print(_variable_descriptive_numerical(variable, data))
        else:
            print(_variable_descriptive_categorical(variable, data))

    def variable_stats_temporal(self, variable):
        """
        Generate a detailed temporal analysis report for a given variable.

        Parameters
        ----------
        name : str
            The variable name.
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        if variable not in self._column_names:
            raise ValueError("variable '{}' not in data.".format(variable))

        if self.date is None:
            raise ValueError("date is not provided. Temporal analysis was "
                             "not run.")

        print(_variable_temporal(variable, self._dict_variables[variable]))

    def variable_plot_descriptive(self, variable, plot_type="pd",
                                  plot_bar_type="event"):
        """
        Plot optimal binning for a given variable.

        Parameters
        ----------
        variable : str
            The variable name.

        plot_type : str (default="pd")
            The measure to show in y-axis. Options are: "pd" and "woe".

        plot_bar_type : str (default="event")
            The count value to show in barplot. Options are: "all", "event"
            and "nonevent".
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        if variable not in self._column_names:
            raise ValueError("variable '{}' not in data.".format(variable))

        optbin = self._get_optbin_variable(variable)
        basic_binning_table = self._dict_variables[variable]["binning_table"]
        basic_splits = self._dict_variables[variable]["splits"]

        _variable_plot_descriptive(optbin, basic_binning_table, basic_splits,
                                   plot_type, plot_bar_type)

    def variable_plot_temporal(self, variable):
        """
        Plot temporal analysis for a given variable.

        Parameters
        ----------
        variable : str
            The variable name.
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        if variable not in self._column_names:
            raise ValueError("variable '{}' not in data.".format(variable))

        if self.date is None:
            raise ValueError("date is not provided. Temporal analysis was "
                             "not run.")

        _variable_plot_temporal(variable, self._dict_variables[variable])

    def _run(self, data, sample_weight=None):
        """Run optimalgrouping (optional), bivariate and decision engine."""
        time_init = time.perf_counter()

        # run optimalgrouping is needed.
        self._run_optimal_grouping(data, sample_weight)

        time_init_metrics = time.perf_counter()

        if self.verbose:
            print("running bivariate analysis...")

        for id, name in enumerate(self._column_names):
            if self.verbose:
                print("\nvariable {}: {}".format(id, name))
            self._run_variable(data[name], name, sample_weight)

        # set actions
        self._decision_engine()

        self._time_run_metrics = time.perf_counter() - time_init_metrics
        self._time_run = time.perf_counter() - time_init

    def _run_optimal_grouping(self, data, sample_weight=None):
        """Run optimal grouping if needed."""
        time_init = time.perf_counter()

        # if optimalgrouping is provided => check status. Run is needed.
        if self.optimalgrouping is None or not self.optimalgrouping._is_run:
            self.optimalgrouping = OptimalGrouping(
                target=self.target, variables_nominal=self.variables_nominal,
                special_values=self.special_values,
                monotonicity_force=self.monotonicity_force,
                special_handler_policy=self.special_handler_policy,
                special_woe_policy=self.special_woe_policy,
                missing_woe_policy=self.missing_woe_policy,
                optbin_options=self.optbin_options,
                verbose=self.verbose)

            # run optimalgrouping and catch exceptions
            try:
                self.optimalgrouping.run(data, sample_weight)
            except EnvironmentError as err:
                print("optimalgrouping - unexpected error: {}".format(
                    sys.exc_info()[0]))
            except Exception as err:
                print("optimalgrouping - error: {}".format(err))
        else:
            if self.verbose:
                print("optimalgrouping: terminated successfully.")

        self._time_run_optimalgrouping = time.perf_counter() - time_init

    def _run_variable(self, data, name, sample_weight=None):
        time_init = time.perf_counter()

        dtype = check_dtype(name, data.values.dtype, self.variables_nominal,
                            self.verbose)

        if self.verbose:
            print("variable dtype: {}".format(dtype))

        if dtype in ("categorical", "nominal"):
            dict_variable_info = self._run_variable_categorical(
                data, name, dtype, sample_weight)
        else:
            dict_variable_info = self._run_variable_numerical(
                data, name, dtype, sample_weight)

        self._dict_variables[name] = dict_variable_info

        dtype = dict_variable_info["dtype"]
        time_run = time.perf_counter() - time_init
        if dtype == "numerical":
            self._time_run_numerical += time_run
        elif dtype == "ordinal":
            self._time_run_ordinal += time_run
        elif dtype == "categorical":
            self._time_run_categorical += time_run
        else:
            self._time_run_nominal += time_run

        return dict_variable_info

    def _run_variable_numerical(self, data, name, dtype, sample_weight=None):
        idx_missing = data.isna()

        if self.special_values:
            idx_special = data.isin(self.special_values)

        # clean data to compute descriptive statistics
        if self.special_values:
            idx_clean = (~idx_special & ~idx_missing)
        else:
            idx_clean = ~idx_missing

        data_clean = data[idx_clean]
        n_informed = len(data_clean)

        if dtype == "numerical" and n_informed < self._n_samples:
            data_int = data_clean.astype(np.int)
            if all(data_clean == data_int):
                data_clean = data_int
                dtype = "ordinal"

                if self.verbose:
                    print("datatype check: variable was casted as ordinal.")

        # prepare data for descriptive bivariate analysis
        y0 = (self._target[idx_clean] == 0)
        y1 = ~y0
        x_event = data_clean[y1]
        x_nonevent = data_clean[y0]

        # generate binning table information using simple buckets
        percentiles_20 = np.linspace(5, 100, 20).astype(np.int)
        splits = weighted_quantile(data_clean, percentiles_20, sample_weight)

        try:
            binning_table = table(values=data.values, target=self._target,
                                  splits=splits, sample_weight=sample_weight,
                                  special_values=self.special_values)[0]
        except Exception as err:
            print("Binning table for variable {} was not computed."
                  "Error: {}.".format(name, str(err)))

        if x_nonevent.size:
            mean_nonevent = np.mean(x_nonevent)
            std_nonevent = np.std(x_nonevent)
            min_nonevent = np.min(x_nonevent)
            max_nonevent = np.max(x_nonevent)
            [percentile_25_nonevent, median_nonevent,
                percentile_75_nonevent] = np.percentile(
                    x_nonevent, [25, 50, 75])
        else:
            [mean_nonevent, std_nonevent, min_nonevent, max_nonevent,
                percentile_25_nonevent, median_nonevent,
                percentile_75_nonevent] = [np.nan] * 7

        if x_event.size:
            mean_event = np.mean(x_event)
            std_event = np.std(x_event)
            min_event = np.min(x_event)
            max_event = np.max(x_event)
            [percentile_25_event, median_event,
                percentile_75_event] = np.percentile(x_event, [25, 50, 75])
        else:
            [mean_event, std_event, min_event, max_event, percentile_25_event,
                median_event, percentile_75_event] = [np.nan] * 7

        # get optbin binning table
        optbin = self._get_optbin_variable(name)
        optbin_binning_table = optbin.binning_table()

        # get IV (optimal) and calculate gini
        iv = optbin._iv_optimal
        gini = _calculate_gini_from_binning_table(optbin_binning_table)

        if x_nonevent.size and x_event.size:
            try:
                # compute p-values
                pvalue_anova = f_oneway(x_nonevent, x_event)[1]
                pvalue_ks = ks_2samp(x_nonevent, x_event)[1]
                pvalue_median_t = median_test(x_nonevent, x_event)[1]
                pvalue_kw = kruskalwallis(x_nonevent, x_event)[1]
            except Exception as err:
                if self.verbose:
                    print("ties of ranked data for variable {} != 0. Error: {}"
                          .format(name, str(err)))

                pvalue_anova = np.nan
                pvalue_ks = 1.0
                pvalue_kw = np.nan
                pvalue_median_t = np.nan

            # compute divergence measure (BBVA - documentation), see [1]
            diff_mean = mean_nonevent - mean_event
            var_nonevent = np.sqrt(std_nonevent)
            var_event = np.sqrt(std_event)
            q = 0.5 * (var_nonevent ** 2 + var_event ** 2)
            divergence = diff_mean ** 2 / q if q else np.nan
        else:
            [pvalue_anova, pvalue_ks, pvalue_kw, pvalue_median_t,
                divergence] = [np.nan] * 5

        # calculate p-value Chi2 and Cramer's V measure of association.
        # generate binning table and extract event and nonevent count
        # columns. Total, special and missing values are removed.
        n_buckets = optbin._n_optimal_buckets
        event = optbin_binning_table["Event count"].values[:n_buckets]
        nonevent = optbin_binning_table["Non-event count"].values[:n_buckets]

        try:
            # compute Chi2 and Cramer's V statistic
            chi2_statistic, pvalue_chi2, _, _ = chi2_contingency(
                [event, nonevent], correction=False)
            pvalue_chi2_max = optbin._max_pvalue
            cramer_v = np.sqrt(chi2_statistic / self._n_samples)
            wasserstein = wasserstein_distance(x_event, x_nonevent)
        except Exception as err:
            if self.verbose:
                print("Chi2 p-value, Cramer's V and Wasserstein distance "
                      "were not computed for variable {}. Error: {}"
                      .format(name, str(err)))

            [pvalue_chi2, pvalue_chi2_max, cramer_v,
                wasserstein] = [np.nan] * 4

        # variable dict
        fields = [
            "dtype", "iv", "gini", "pvalue_chi2", "pvalue_chi2_max",
            "cramer_v", "splits", "binning_table", "pvalue_anova", "pvalue_ks",
            "pvalue_kw", "pvalue_median_t", "divergence",
            "wasserstein_distance", "mean_nonevent", "std_nonevent",
            "min_nonevent", "percentile_25_nonevent", "median_nonevent",
            "percentile_75_nonevent", "max_nonevent", "mean_event",
            "std_event", "min_event", "percentile_25_event", "median_event",
            "percentile_75_event", "max_event", "pd_monotonicity",
            "groups", "group_special", "group_missing", "group_others",
            "optimal_binning_table"]

        info = [
            dtype, iv, gini, pvalue_chi2, pvalue_chi2_max, cramer_v,
            splits, binning_table, pvalue_anova, pvalue_ks, pvalue_kw,
            pvalue_median_t, divergence, wasserstein, mean_nonevent,
            std_nonevent, min_nonevent, percentile_25_nonevent,
            median_nonevent, percentile_75_nonevent, max_nonevent, mean_event,
            std_event, min_event, percentile_25_event, median_event,
            percentile_75_event, max_event, optbin.monotonicity_sense,
            optbin._n_optimal_buckets, optbin._group_special,
            optbin._group_missing, optbin._group_others, optbin_binning_table]

        dict_variable_info = dict(zip(fields, info))

        # temporal analysis
        if self.date is not None:
            dict_variable_t_info = self._run_temporal(data, name, dtype)
            dict_variable_info = {**dict_variable_info, **dict_variable_t_info}

        return dict_variable_info

    def _run_variable_categorical(self, data, name, dtype, sample_weight=None):
        idx_missing = data.isna()

        if sample_weight is None:
            sample_weight = np.ones(data.shape[0])

        if self.special_values:
            idx_special = data.isin(self.special_values)

        # clean data to compute descriptive statistics
        if self.special_values:
            idx_clean = (~idx_special & ~idx_missing)
        else:
            idx_clean = ~idx_missing

        data_clean = data[idx_clean]
        sample_weight_clean = sample_weight[idx_clean]
        n_informed = sum(sample_weight_clean)

        if dtype == "categorical":
            data_clean = data_clean.astype(str)

        # number of categories
        unique_categories = weighted_value_counts(
            data_clean, sample_weight_clean)
        u_categories = unique_categories.index.values
        n_u_categories = unique_categories.values
        n_categories = len(u_categories)
        most_freq_category = u_categories[0]
        p_most_freq_category = n_u_categories[0] / n_informed

        # top 10 most frequent categories
        max_categories = min(n_categories, 10)
        top_categories = u_categories[:max_categories]
        n_top_categories = n_u_categories[:max_categories]

        others = [c for c in u_categories if c not in top_categories]
        splits = [[c] for c in top_categories] + [others]

        # generate binning table with top categories and others
        try:
            binning_table = table(values=data, target=self._target,
                                  splits=splits, sample_weight=sample_weight,
                                  special_values=self.special_values)[0]
        except Exception as err:
            print("Binning table for variable {} was not computed."
                  "Error: {}.".format(name, str(err)))

        # get optbin binning table
        optbin = self._get_optbin_variable(name)
        optbin_binning_table = optbin.binning_table()

        # get IV (optimal) and calculate gini
        iv = optbin._iv_optimal
        gini = _calculate_gini_from_binning_table(optbin_binning_table)

        # distribution of number of categories
        n_categories_split = np.array([len(x) for x in optbin.splits_optimal])

        if n_categories_split.size:
            mean_n_categories = np.mean(n_categories_split)
            std_n_categories = np.std(n_categories_split)
            min_n_categories = np.min(n_categories_split)
            max_n_categories = np.max(n_categories_split)

            [percentile_25_n_categories, median_n_categories,
                percentile_75_n_categories] = np.percentile(
                    n_categories_split, [25, 50, 75])
        else:
            [mean_n_categories, std_n_categories, min_n_categories,
                max_n_categories, percentile_25_n_categories,
                median_n_categories, percentile_75_n_categories] = [np.nan] * 7

        # calculate p-value Chi2 and Cramer's V measure of association.
        # generate binning table and extract event and nonevent count
        # columns. Total, special and missing values are removed.
        n_buckets = optbin._n_optimal_buckets
        event = optbin_binning_table["Event count"].values[:n_buckets]
        nonevent = optbin_binning_table["Non-event count"].values[:n_buckets]

        try:
            # compute Chi2 and Cramer's V statistic
            chi2_statistic, pvalue_chi2, _, _ = chi2_contingency(
                [event, nonevent], correction=False)

            pvalue_chi2_max = optbin._max_pvalue
            cramer_v = np.sqrt(chi2_statistic / self._n_samples)
        except Exception as err:
            if self.verbose:
                print("Chi2 p-value, Cramer's V and were not computed for "
                      "variable {}. Error: {}".format(name, str(err)))
            [pvalue_chi2, pvalue_chi2_max, cramer_v] = [np.nan] * 3

        # variable dict
        fields = [
            "dtype", "iv", "gini", "pvalue_chi2", "pvalue_chi2_max",
            "cramer_v", "splits", "binning_table", "categories",
            "n_categories", "most_freq_category", "p_most_freq_category",
            "top_categories", "n_top_categories", "mean_n_categories",
            "std_n_categories", "min_n_categories",
            "percentile_25_n_categories", "median_n_categories",
            "percentile_75_n_categories", "max_n_categories",
            "pd_monotonicity", "groups", "group_special", "group_missing",
            "group_others", "optimal_binning_table"]

        info = [
            dtype, iv, gini, pvalue_chi2, pvalue_chi2_max, cramer_v, splits,
            binning_table, u_categories, n_categories, most_freq_category,
            p_most_freq_category, top_categories, n_top_categories,
            mean_n_categories, std_n_categories, min_n_categories,
            percentile_25_n_categories, median_n_categories,
            percentile_75_n_categories, max_n_categories,
            optbin.monotonicity_sense, optbin._n_optimal_buckets,
            optbin._group_special, optbin._group_missing, optbin._group_others,
            optbin_binning_table]

        dict_variable_info = dict(zip(fields, info))

        # temporal analysis
        if self.date is not None:
            dict_variable_t_info = self._run_temporal(data, name, dtype)
            dict_variable_info = {**dict_variable_info, **dict_variable_t_info}

        return dict_variable_info

    def _run_temporal(self, data, name, dtype):
        # get optbin variable
        optbin = self._get_optbin_variable(name)

        # optimal splits
        optbin_splits = optbin.splits_optimal
        n_splits = len(optbin_splits)

        # initialize woe and groups arrays
        t_split = np.zeros(self._n_samples)

        if not n_splits:
            t_split = np.zeros(self._n_samples)
            idx_nan = data.isna()
            # a single category increase n_splits for special and missing
            n_splits = 1
        elif dtype in ("categorical", "nominal"):
            # categorical and nominal variables return groups as
            # numpy.ndarray objects.
            for _idx, split in enumerate(optbin_splits):
                mask = data.isin(split).values
                t_split[mask] = _idx
            idx_nan = data.isna()
        else:
            # numerical and ordinal variables return extra group
            # (> last split)
            splits = optbin_splits[::-1]
            x = data.values
            mask = (x > splits[-1])
            t_split[mask] = n_splits
            for _idx, split in enumerate(splits):
                mask = (x <= split)
                t_split[mask] = n_splits - (_idx + 1)
            # indexes with NaN n x
            idx_nan = data.isna()
            # account for > group
            n_splits += 1

        # special values
        if optbin._splits_specials:
            x = data.values
            for _idx, split in enumerate(optbin._splits_specials):
                if isinstance(split, np.ndarray):
                    mask = data.isin(split).values
                else:
                    mask = (x == split)
                t_split[mask] = n_splits
                n_splits += 1
        else:
            n_splits += 1

        # missing values
        t_split[idx_nan] = n_splits

        # filter by unique date and store occurrences and event information
        n_unique_dates = len(self._unique_dates)
        split_count_by_date = np.zeros((n_splits+1, n_unique_dates))
        split_event_by_date = np.zeros((n_splits+1, n_unique_dates))

        for id_date, udate in enumerate(self._unique_dates):
            mask_date = (self._dates == udate)

            for split in range(n_splits + 1):
                mask_split = (t_split == split)
                mask_split_date = (mask_date & mask_split)
                split_count_by_date[split, id_date] = np.count_nonzero(
                    mask_split_date)
                split_event_by_date[split, id_date] = np.count_nonzero(
                    self._target[mask_split_date] == 1)

        t_split_count_by_date = split_count_by_date
        t_split_event_by_date = split_event_by_date
        t_n_split_empty = np.count_nonzero(split_count_by_date == 0, axis=0)
        t_n_months_no_event = np.count_nonzero(
            split_event_by_date == 0, axis=0)

        # event rate temporal statistics
        split_pd = split_event_by_date / split_count_by_date
        t_pd_mean = np.mean(split_pd, axis=1)
        t_pd_std = np.std(split_pd, axis=1)
        t_pd_min = np.min(split_pd, axis=1)
        t_pd_max = np.max(split_pd, axis=1)
        [t_pd_percentile_25, t_pd_median,
            t_pd_percentile_75] = np.percentile(split_pd, [25, 50, 75], axis=1)
        iqr_n = t_pd_percentile_75 - t_pd_percentile_25
        iqr_d = t_pd_percentile_75 + t_pd_percentile_25
        t_pd_dispersion = iqr_n / iqr_d

        # IV
        iv_date = np.zeros(n_unique_dates)
        for id_date in range(n_unique_dates):
            event = split_event_by_date[:, id_date]
            nonevent = split_count_by_date[:, id_date] - event
            event_p = event / event.sum()
            nonevent_p = nonevent / nonevent.sum()
            iv_sum = 0
            for i in range(len(event)):
                if nonevent_p[i] * event_p[i]:
                    woe = np.log(nonevent_p[i] / event_p[i])
                    iv_sum += woe * (nonevent_p[i] - event_p[i])
            iv_date[id_date] = iv_sum

        t_iv_by_date = iv_date
        t_iv_mean = np.nanmean(t_iv_by_date)
        t_iv_std = np.nanstd(t_iv_by_date)
        t_iv_min = np.nanmin(t_iv_by_date)
        t_iv_max = np.nanmax(t_iv_by_date)
        [t_iv_percentile_25, t_iv_median,
            t_iv_percentile_75] = np.nanpercentile(t_iv_by_date, [25, 50, 75])

        # divergence
        split_date = split_count_by_date / np.sum(split_count_by_date, axis=0)
        t_split_div_uniform = js_divergence_multivariate(split_date, "uniform")
        t_split_div_exponential = js_divergence_multivariate(
            split_date, "exponential")

        fields = [
            "t_split_count_by_date", "t_split_event_by_date",
            "t_n_split_empty", "t_n_months_no_event", "t_pd_mean", "t_pd_std",
            "t_pd_min", "t_pd_percentile_25", "t_pd_median",
            "t_pd_percentile_75", "t_pd_max", "t_pd_dispersion", "t_iv_mean",
            "t_iv_std", "t_iv_min", "t_iv_percentile_25", "t_iv_median",
            "t_iv_percentile_75", "t_iv_max", "t_split_div_uniform",
            "t_split_div_exponential"]

        info = [
            t_split_count_by_date, t_split_event_by_date,
            t_n_split_empty, t_n_months_no_event, t_pd_mean, t_pd_std,
            t_pd_min, t_pd_percentile_25, t_pd_median,
            t_pd_percentile_75, t_pd_max, t_pd_dispersion, t_iv_mean,
            t_iv_std, t_iv_min, t_iv_percentile_25, t_iv_median,
            t_iv_percentile_75, t_iv_max, t_split_div_uniform,
            t_split_div_exponential]

        return dict(zip(fields, info))

    def _transform_remove(self, data, mode):
        """Remove unnecessary variables in-place."""
        AGGR_COMMENT = "auto: remove due to aggressive mode"
        BASIC_COMMENT = "auto: keep due to basic mode"

        remove = [column for column in self._column_names if
                  self._dict_variables[column]["action"] == "remove"]

        review = [column for column in self._column_names if
                  self._dict_variables[column]["action"] == "review"]

        if mode == "aggressive":
            if self.verbose:
                print("columns with action='remove' or action='review' "
                      "except the target and date column if supplied are "
                      "dropped.")

            for column in review:
                remove.append(column)
                self._dict_variables[column]["action"] = "remove"
                self._dict_variables[column]["auto_comment"] = AGGR_COMMENT

        elif mode == "basic":
            if self.verbose:
                print("only columns with action='remove' are dropped.")

            # if recommended action is not removed then keep
            for column in review:
                self._dict_variables[column]["action"] = "keep"
                self._dict_variables[column]["auto_comment"] = BASIC_COMMENT

        self._n_vars_remove = len(remove)

        # drop variables in-place
        data.drop(remove, axis=1, inplace=True)

        # garbage collector
        gc.collect()

    def _decision_engine(self):
        """Set action to be taken for each variable."""
        MSG_METRIC_HIGH = "{} value above threshold {:.3f} > {:.3f}"
        MSG_METRIC_LOW = "{} value below threshold {:.3f} < {:.3f}"
        MSG_UNDEFINED = "undefined monotonicity"

        for column in self._column_names:
            # initialize statuses and comments
            self._dict_variables[column]["status"] = []
            self._dict_variables[column]["comment"] = []

            # IV flags
            if self._dict_variables[column]["iv"] < self.min_iv:
                self._dict_variables[column]["status"].append(STATUS_IV_LOW)
                self._dict_variables[column]["comment"].append(
                    MSG_METRIC_LOW.format(
                        "IV", self._dict_variables[column]["iv"], self.min_iv))
            elif self._dict_variables[column]["iv"] > self.max_iv:
                self._dict_variables[column]["status"].append(STATUS_IV_HIGH)
                self._dict_variables[column]["comment"].append(
                    MSG_METRIC_HIGH.format(
                        "IV", self._dict_variables[column]["iv"], self.max_iv))

            # gini flags
            if self._dict_variables[column]["gini"] < self.min_gini:
                self._dict_variables[column]["status"].append(STATUS_GINI_LOW)
                self._dict_variables[column]["comment"].append(
                    MSG_METRIC_LOW.format(
                        "gini", self._dict_variables[column]["gini"],
                        self.min_gini))
            elif self._dict_variables[column]["gini"] > self.max_gini:
                self._dict_variables[column]["status"].append(STATUS_GINI_HIGH)
                self._dict_variables[column]["comment"].append(
                    MSG_METRIC_HIGH.format(
                        "gini", self._dict_variables[column]["gini"],
                        self.max_gini))

            # p-value chi2 and Cramer's V
            if (self._dict_variables[column]["pvalue_chi2"]
                    > self.max_pvalue_chi2):
                self._dict_variables[column]["status"].append(
                    STATUS_PVALUE_CHI2_HIGH)
                self._dict_variables[column]["comment"].append(
                    MSG_METRIC_HIGH.format(
                        "pvalue_chi2",
                        self._dict_variables[column]["pvalue_chi2"],
                        self.max_pvalue_chi2))

            if (self._dict_variables[column]["pvalue_chi2_max"]
                    > self.max_pvalue_chi2):
                self._dict_variables[column]["status"].append(
                    STATUS_PVALUE_MAX_CHI2_HIGH)
                self._dict_variables[column]["comment"].append(
                    MSG_METRIC_HIGH.format(
                        "pvalue_chi2_max",
                        self._dict_variables[column]["pvalue_chi2_max"],
                        self.max_pvalue_chi2))

            if self._dict_variables[column]["cramer_v"] < self.min_cramer_v:
                self._dict_variables[column]["status"].append(
                    STATUS_CRAMER_LOW)
                self._dict_variables[column]["comment"].append(
                    MSG_METRIC_LOW.format(
                        "cramer_v",
                        self._dict_variables[column]["cramer_v"],
                        self.min_cramer_v))

            # optimalgrouping undefined
            optbin = self._get_optbin_variable(column)
            if optbin.monotonicity_sense == "undefined":
                self._dict_variables[column]["status"].append(STATUS_UNDEFINED)
                self._dict_variables[column]["comment"].append(MSG_UNDEFINED)

            if self._dict_variables[column]["dtype"] in ("numerical",
                                                         "ordinal"):
                if (self._dict_variables[column]["pvalue_anova"]
                        > self.max_pvalue_anova):
                    self._dict_variables[column]["status"].append(
                        STATUS_PVALUE_ANOVA_HIGH)
                    self._dict_variables[column]["comment"].append(
                        MSG_METRIC_HIGH.format(
                            "pvalue_anova",
                            self._dict_variables[column]["pvalue_anova"],
                            self.max_pvalue_anova))

                if (self._dict_variables[column]["pvalue_ks"]
                        > self.max_pvalue_ks):
                    self._dict_variables[column]["status"].append(
                        STATUS_PVALUE_KS_HIGH)
                    self._dict_variables[column]["comment"].append(
                        MSG_METRIC_HIGH.format(
                            "pvalue_ks",
                            self._dict_variables[column]["pvalue_ks"],
                            self.max_pvalue_ks))

                if (self._dict_variables[column]["pvalue_kw"]
                        > self.max_pvalue_kw):
                    self._dict_variables[column]["status"].append(
                        STATUS_PVALUE_KW_HIGH)
                    self._dict_variables[column]["comment"].append(
                        MSG_METRIC_HIGH.format(
                            "pvalue_kw",
                            self._dict_variables[column]["pvalue_kw"],
                            self.max_pvalue_kw))

                if (self._dict_variables[column]["pvalue_median_t"]
                        > self.max_pvalue_median_t):
                    self._dict_variables[column]["status"].append(
                        STATUS_PVALUE_MEDIAN_T_HIGH)
                    self._dict_variables[column]["comment"].append(
                        MSG_METRIC_HIGH.format(
                            "pvalue_median_t",
                            self._dict_variables[column]["pvalue_median_t"],
                            self.max_pvalue_median_t))

                if not self._dict_variables[column]["status"]:
                    self._dict_variables[column]["status"].append(STATUS_OK)

            status = self._dict_variables[column]["status"]
            if any(st in STATUS_REMOVE for st in status):
                self._dict_variables[column]["action"] = "remove"
            elif any(st in STATUS_REVIEW for st in status):
                self._dict_variables[column]["action"] = "review"
            else:
                self._dict_variables[column]["action"] = "keep"

            action = self._dict_variables[column]["action"]
            self._dict_variables[column]["recommended_action"] = action

            if self.optbin_options is not None:
                if column in self.optbin_options.keys():
                    # pass optbin options
                    user_params = self.optbin_options[column]["params"]
                    user_comment = self.optbin_options[column]["comment"]

                    self._dict_variables[column][
                        "optbin_params"] = user_params
                    self._dict_variables[column][
                        "optbin_comment"] = user_comment

                    # pass optbin options differences
                    opt_d = self.optimalgrouping._dict_optbin_options[column]
                    self._dict_variables[column]["optbin_default"] = opt_d

    def _get_optbin_variable(self, name):
        variable_id = self.optimalgrouping._detect_name_id(name)
        return self.optimalgrouping._variables_information[variable_id]
