"""
Exploratory Data Analysis (EDA) - univariate analysis
"""

# Authors:
#   Guillermo Navas-Palencia <guillermo.navas@bbva.com>
#   Luis Tarrega Ruiz <luis.tarrega@bbva.com>
# BBVA - Copyright 2019.

import gc
import numbers
import time

import numpy as np
import pandas as pd

from ..core.base import GRMlabProcess
from ..core.dtypes import check_date_format
from ..core.dtypes import check_dtype
from ..core.dtypes import check_target_dtype
from ..core.exceptions import NotRunException
from ..reporting.util import reporting_output_format
from .util import js_divergence_multivariate


STATUS_MISSING = "missing"
STATUS_SPECIAL = "special"
STATUS_DATA_DIVERGENCE = "data_divergence"
STATUS_INFO_DIVERGENCE = "info_divergence"
STATUS_OK = "ok"

STATUS_OPTIONS = [
    STATUS_MISSING, STATUS_SPECIAL, STATUS_DATA_DIVERGENCE,
    STATUS_INFO_DIVERGENCE, STATUS_OK]

STATUS_REMOVE = [STATUS_MISSING, STATUS_SPECIAL]

STATUS_REVIEW = [STATUS_DATA_DIVERGENCE, STATUS_INFO_DIVERGENCE]


def _quality_score(p_missing, p_special, t_info_div=None, t_data_div=None):
    p_informed = 1.0 - (p_missing + p_special)
    # main = entropy(np.array([p_missing, p_special, p_informed])) / np.log(3)
    main = p_informed

    if t_info_div is not None and t_data_div is not None:
        rest = np.exp(np.array([t_info_div, t_data_div]).mean() - 1.0)

        return main * rest
    else:
        return main


def _univariate_results(dict_variables, step, date, format):
    """Return run or transform step results as a dataframe or json."""

    # common metrics
    common_cols = [
        "name", "dtype", "recommended_action", "status", "comment",
        "n_missing", "p_missing", "n_special", "p_special", "u_special",
        "quality_score"]

    # numerical/ordinal metrics
    num_cols = [
        "d_n_zero", "d_p_zero", "d_n_neg", "d_n_pos", "d_min", "d_max",
        "d_mean", "d_std", "d_mode", "d_percentile_1", "d_percentile_25",
        "d_median", "d_percentile_75", "d_percentile_99", "d_outlier_low",
        "d_outlier_high", "d_n_outlier_low", "d_n_outlier_high",
        "d_coeff_disp_param", "d_coeff_disp_nonparam", "d_hist_cols",
        "d_hist_pos", "d_concentration", "d_concentration_interval"]

    # categorical/nominal metrics
    cat_cols = [
        "d_n_categories", "d_most_freq_category", "d_p_most_freq_category",
        "d_top_categories", "d_n_top_categories", "d_hist_cols", "d_hist_pos",
        "d_hhi", "d_hhi_norm"]

    # transform
    order_cols_transform = [
        "name", "dtype", "action", "status", "comment", "recommended_action",
        "user_action", "user_comment", "auto_comment"]

    if date:
        temporal_cols = [
            "t_n_months_no_info", "t_info_div_uniform",
            "t_info_div_exponential", "t_data_div_uniform",
            "t_data_div_exponential"]

        num_order_cols = common_cols + num_cols + temporal_cols
        cat_order_cols = common_cols + cat_cols + temporal_cols
    else:
        num_order_cols = common_cols + num_cols
        cat_order_cols = common_cols + cat_cols

    results = pd.DataFrame.from_dict(dict_variables).T
    results.reset_index(level=0, inplace=True)
    results.rename(columns={"index": "name"}, inplace=True)

    if step == "run":
        if "comment" not in results.columns:
            results["comment"] = np.nan

        if all([c in results.columns for c in num_cols]):
            num_results = results[num_order_cols][results.dtype.isin(
                ["numerical", "ordinal"])]

            num_report = reporting_output_format(num_results, format)
        else:
            num_results = pd.DataFrame(columns=num_order_cols)
            num_report = reporting_output_format(num_results, format)

        if all([c in results.columns for c in cat_cols]):
            cat_results = results[cat_order_cols][results.dtype.isin(
                ["categorical", "nominal"])]

            cat_report = reporting_output_format(cat_results, format)
        else:
            cat_results = pd.DataFrame(columns=cat_order_cols)
            cat_report = reporting_output_format(cat_results, format)

        return num_report, cat_report
    else:
        for col in ["comment", "user_action", "user_comment", "auto_comment"]:
            if col not in results.columns:
                results[col] = np.nan

        results = results[order_cols_transform]
        return reporting_output_format(results, format)


def _univariate_stats(report_data, step):
    if step == "run":
        report = (
            "\033[94m================================================================================\033[0m\n"
            "\033[1m\033[94m                           GRMlab Univariate 0.1: Run                          \033[0m\n"
            "\033[94m================================================================================\033[0m\n"
            "\n"
            " \033[1mGeneral information                      Configuration options\033[0m\n"
            "   number of samples   {:>8}             special values                {:>3}\n"
            "   number of variables {:>8}             nominal variables             {:>3}\n"
            "   target variable     {:>8}\n"
            "   target dtype        {:>8}\n"
            "   date variable       {:>8}\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            "\n"
            " \033[1mVariables analysis                       Findings analysis\033[0m\n"
            "   numerical           {:>8}             high % missing               {:>4}\n"
            "   ordinal             {:>8}             high % special               {:>4}\n"
            "   categorical         {:>8}             high divergence info         {:>4}\n"
            "   nominal             {:>8}             high divergence data         {:>4}\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            "\n"
            " \033[1mTiming statistics\033[0m\n"
            "   total             {:>6.3f}\n"
            "     numerical       {:>6.3f} ({:>5.1%})\n"
            "     ordinal         {:>6.3f} ({:>5.1%})\n"
            "     categorical     {:>6.3f} ({:>5.1%})\n"
            "     nominal         {:>6.3f} ({:>5.1%})\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            )
    else:
        report = (
            "\033[94m================================================================================\033[0m\n"
            "\033[1m\033[94m                          GRMlab Univariate 0.1: Transform                          \033[0m\n"
            "\033[94m================================================================================\033[0m\n"
            "\n"
            " \033[1mResults                                        Timing statistics\033[0m\n"
            "   original data       {:>8}                   total        {:>6.3f}\n"
            "   after univariate    {:>8}\n"
            "   removed             {:>8}\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            "\n"
            "\033[1mUser actions\033[0m\n"
            "   number of actions   {:>8}\n"
            "   number of comments  {:>8} ({:>5.1%})\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            )

    return report.format(*report_data)


def _temporal_numerical(data, data_clean, dates, dates_clean, u_dates,
                        special_values):
    """Temporal analysis for numerical and ordinal variables."""
    t_missing = []
    t_special = []
    t_info = []

    t_probabilities = []
    t_mean = []
    t_p25 = []
    t_median = []
    t_p75 = []

    find_reference = False
    for date in u_dates[::-1]:
        u_date_data = data_clean[dates_clean == date]
        if u_date_data.size:
            reference = np.percentile(u_date_data, list(range(10, 100, 10)))
            find_reference = True
            break

    if not find_reference:
        n_months_no_info = len(u_dates)
    else:
        n_months_no_info = 0

        for date in u_dates:
            u_date_data = data[dates == date]
            n_u_date_missing = np.count_nonzero(pd.isnull(u_date_data))
            n_u_date_special = np.count_nonzero(
                pd.Series(u_date_data).isin(special_values))
            n_u_date_issues = n_u_date_missing + n_u_date_special
            n_u_date_informed = u_date_data.size - n_u_date_issues

            n_u_date_data = len(u_date_data)
            t_missing.append(n_u_date_missing / n_u_date_data)
            t_special.append(n_u_date_special / n_u_date_data)
            t_info.append(n_u_date_informed / n_u_date_data)

            u_date_data = data_clean[dates_clean == date]
            if u_date_data.size:
                [p25, median, p75] = np.percentile(u_date_data, [25, 50, 75])
                mean = np.mean(u_date_data)
                proba = [np.count_nonzero(u_date_data <= r) / len(u_date_data)
                         for r in reference]
            else:
                n_months_no_info += 1
                [mean, p25, median, p75] = [np.nan, np.nan, np.nan, np.nan]
                proba = [np.nan] * 9

            t_mean.append(mean)
            t_p25.append(p25)
            t_median.append(median)
            t_p75.append(p75)
            t_probabilities.append(proba)

    # compute divergence measures
    X_info = np.asarray([t_info, t_special, t_missing])
    t_info_div_uniform = js_divergence_multivariate(X_info, "uniform")
    t_info_div_exponential = js_divergence_multivariate(X_info, "exponential")

    X_data = np.asarray(t_probabilities)
    t_data_div_uniform = js_divergence_multivariate(X_data.T, "uniform")
    t_data_div_exponential = js_divergence_multivariate(
        X_data.T, "exponential")

    fields = [
        "t_mean", "t_p25", "t_median", "t_p75", "t_n_months_no_info",
        "t_missing", "t_special", "t_info", "t_info_div_uniform",
        "t_info_div_exponential", "t_data_div_uniform",
        "t_data_div_exponential"]

    info = [
        t_mean, t_p25, t_median, t_p75, n_months_no_info, t_missing,
        t_special, t_info, t_info_div_uniform, t_info_div_exponential,
        t_data_div_uniform, t_data_div_exponential]

    return dict(zip(fields, info))


def _temporal_categorical(data, data_clean, dates, dates_clean, u_dates,
                          special_values, d_top_categories):
    """Temporal analysis for categorical and nominal variables."""
    t_missing = []
    t_special = []
    t_info = []

    max_categories = 10
    t_dict_categories = dict(zip(np.arange(0, max_categories + 1),
                             [[] for c in range(max_categories + 1)]))

    t_probabilities = []

    n_months_no_info = 0

    for date in u_dates:
        u_date_data = data[dates == date]
        n_u_date_missing = np.count_nonzero(pd.isnull(u_date_data))
        n_u_date_special = np.count_nonzero(
            pd.Series(u_date_data).isin(special_values))
        n_u_date_issues = n_u_date_missing + n_u_date_special
        n_u_date_informed = u_date_data.size - n_u_date_issues

        n_u_date_data = len(u_date_data)
        t_missing.append(n_u_date_missing / n_u_date_data)
        t_special.append(n_u_date_special / n_u_date_data)
        t_info.append(n_u_date_informed / n_u_date_data)

        u_date_data = data_clean[dates_clean == date]
        if u_date_data.size:
            proba_categories = []
            for idx, category in enumerate(d_top_categories):
                n_u_date = len(u_date_data)
                proba = np.count_nonzero(u_date_data == category) / n_u_date
                t_dict_categories[idx].append(proba)
                proba_categories.append(proba)

            t_probabilities.append(proba_categories)
            rest = round(1.0 - np.sum(proba_categories), 8)
            t_dict_categories[max_categories].append(rest)
        else:
            n_months_no_info += 1
            for idx, category in enumerate(d_top_categories):
                t_dict_categories[idx].append(np.nan)
            t_dict_categories[max_categories].append(np.nan)

    # compute divergence measures
    X_info = np.asarray([t_info, t_special, t_missing])
    t_info_div_uniform = js_divergence_multivariate(X_info, "uniform")
    t_info_div_exponential = js_divergence_multivariate(
        X_info, "exponential")

    X_data = np.asarray(t_probabilities)
    t_data_div_uniform = js_divergence_multivariate(X_data.T, "uniform")
    t_data_div_exponential = js_divergence_multivariate(
        X_data.T, "exponential")

    fields = [
        "t_n_months_no_info", "t_missing", "t_special", "t_info",
        "t_info_div_uniform", "t_info_div_exponential", "t_data_div_uniform",
        "t_data_div_exponential"]

    info = [
        n_months_no_info, t_missing, t_special, t_info, t_info_div_uniform,
        t_info_div_exponential, t_data_div_uniform, t_data_div_exponential]

    fields_categories = ["t_c{}".format(i) for i in range(max_categories)]
    fields_categories.append("t_rest")

    fields += fields_categories
    info += t_dict_categories.values()

    return dict(zip(fields, info))


def _information_metrics(data, n_samples, special_values):
    # special values
    if special_values:
        idx_special = data.isin(special_values)
        n_special = np.count_nonzero(idx_special)
        p_special = n_special / n_samples
        u_special = data[idx_special].unique()
        n_u_special = len(u_special)
    else:
        idx_special = None
        n_special = 0
        p_special = 0
        u_special = []
        n_u_special = 0

    # missing values
    idx_missing = data.isna()
    n_missing = np.count_nonzero(idx_missing)
    p_missing = n_missing / n_samples

    # informed values
    n_informed = n_samples - n_missing - n_special

    return [idx_special, n_special, p_special, u_special, n_u_special,
            idx_missing, n_missing, p_missing, n_informed]


class Univariate(GRMlabProcess):
    """
    Univariate data analysis.

    Perform basic exploratory data analysis on a dataset.

    Parameters
    ----------
    target : str or None (default=None)
        The name of the variable flagged as target.

    date : str or None (default=None)
        The name of the variable flagged as date.

    variables_nominal : list or None (default=None)
        List of ordinal variables to be treated as nominal.

    special_values : list or None (default=None)
        List of special values to be considered.

    max_p_missing : float (default=0.99)
        Threshold maximum percentage of missing values per variable.

    max_p_special : float (default=0.99)
        Threshold maximum percentage of special values per variable.

    max_divergence : float (default=0.2)
        Threshold maximum divergence measure value.

    verbose : int or boolean (default=False)
        Controls verbosity of output.
    """
    def __init__(self, target=None, date=None, variables_nominal=None,
                 special_values=None, max_p_missing=0.99, max_p_special=0.99,
                 max_divergence=0.2, verbose=False):

        self.target = target
        self.date = date

        if variables_nominal is None:
            self.variables_nominal = []
        else:
            self.variables_nominal = variables_nominal

        if special_values is None:
            self.special_values = []
        else:
            self.special_values = special_values

        self.verbose = verbose

        self.max_p_missing = max_p_missing
        self.max_p_special = max_p_special
        self.max_divergence = max_divergence

        self._n_samples = None
        self._n_vars = None
        self._n_vars_remove = None

        self._dates = []
        self._unique_dates = []

        self._target_dtype = None

        self._column_names = []
        self._dict_variables = {}

        self._time_run = None
        self._time_run_numerical = 0
        self._time_run_ordinal = 0
        self._time_run_nominal = 0
        self._time_run_categorical = 0
        self._time_transform = None

        # transform parameters
        self._transform_mode = None

        # flags
        self._is_run = False
        self._is_transformed = False

    def results(self, step="run", format="dataframe"):
        """
        Return information and flags for each variable.

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
        return _univariate_results(self._dict_variables, step, date, format)

    def run(self, data):
        """
        Run univariate analysis.

        Note that temporal analysis is only performed if a date variable is
        provided during instantiation.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw dataset.

        Returns
        -------
        self : object
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas dataframe.")

        self._n_samples = len(data)
        self._n_vars = len(data.columns)
        # variable names to lowercase
        data.columns = map(str.lower, data.columns)
        self._column_names = list(data.columns)

        if not isinstance(self.max_p_missing, numbers.Number) or (
                self.max_p_missing < 0.0 or self.max_p_missing > 1.0):
            raise ValueError("max_p_missing must be a positive number "
                             "in [0, 1].".format(self.max_p_missing))

        if not isinstance(self.max_p_special, numbers.Number) or (
                self.max_p_special < 0.0 or self.max_p_special > 1.0):
            raise ValueError("max_p_special must be a positive number "
                             "in [0, 1].".format(self.max_p_special))

        if not isinstance(self.max_divergence, numbers.Number) or (
                self.max_divergence < 0.0 or self.max_divergence > 1.0):
            raise ValueError("max_divergence must be a positive number "
                             "in [0, 1].".format(self.max_divergence))

        if self.target is not None:
            self.target = self.target.lower()
        if self.date is not None:
            self.date = self.date.lower()

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

        # target information
        if self.target is not None:
            self._target_dtype = check_target_dtype(data[self.target].values)
            self._column_names.remove(self.target)

        # run univariate
        self._run(data)

        self._is_run = True

        return self

    def stats(self, step="run"):
        """
        Univariate analysis statistics.

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
            target_flag = self.target if self.target is not None else "not set"

            if self._target_dtype is not None:
                target_dtype_flag = self._target_dtype
            else:
                target_dtype_flag = "not set"

            date_flag = self.date if self.date is not None else "not set"

            # timing
            perc_time_numerical = self._time_run_numerical / self._time_run
            perc_time_ordinal = self._time_run_ordinal / self._time_run
            perc_time_categorical = self._time_run_categorical / self._time_run
            perc_time_nominal = self._time_run_nominal / self._time_run

            [n_numerical, n_ordinal, n_categorical, n_nominal] = [
                sum(d["dtype"] == dtype for d in dict_values) for dtype
                in ("numerical", "ordinal", "categorical", "nominal")]

            hp_missing = sum(d["status"] == STATUS_MISSING
                             for d in dict_values)
            hp_special = sum(d["status"] == STATUS_SPECIAL
                             for d in dict_values)

            if self.date is not None:
                hp_div_info = sum(d["status"] == STATUS_INFO_DIVERGENCE
                                  for d in dict_values)
                hp_div_data = sum(d["status"] == STATUS_DATA_DIVERGENCE
                                  for d in dict_values)
            else:
                hp_div_info = 0
                hp_div_data = 0

            # prepare data
            report_data = [
                self._n_samples, special_flag, self._n_vars,
                nominal_flag, target_flag, target_dtype_flag, date_flag,
                n_numerical, hp_missing, n_ordinal, hp_special, n_categorical,
                hp_div_info, n_nominal, hp_div_data, self._time_run,
                self._time_run_numerical, perc_time_numerical,
                self._time_run_ordinal, perc_time_ordinal,
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

        print(_univariate_stats(report_data, step))

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

        return self

    def _run(self, data):
        """Run univariate and decision engine."""
        time_init = time.perf_counter()

        if self.verbose:
            print("running univariate analysis...")

        for id, name in enumerate(self._column_names):
            if self.verbose:
                print("\nvariable {}: {}".format(id, name))
            self._run_variable(data[name], name)

        # set actions
        self._decision_engine()

        self._time_run = time.perf_counter() - time_init

    def _run_variable(self, data, name):
        time_init = time.perf_counter()

        dtype = check_dtype(name, data.values.dtype, self.variables_nominal,
                            self.verbose)

        if self.verbose:
            print("variable dtype: {}".format(dtype))

        if dtype in ("categorical", "nominal"):
            dict_variable_info = self._run_variable_categorical(data, dtype)
        else:
            dict_variable_info = self._run_variable_numerical(data, dtype)

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

    def _run_variable_numerical(self, data, dtype):
        [idx_special, n_special, p_special, u_special, n_u_special,
            idx_missing, n_missing, p_missing,
            n_informed] = _information_metrics(
                data, self._n_samples, self.special_values)

        # clean data to compute descriptive statistics
        if self.special_values:
            idx_clean = (~idx_special & ~idx_missing)
        else:
            idx_clean = ~idx_missing

        data_clean = data[idx_clean]

        if dtype == "numerical" and n_informed < self._n_samples:
            data_int = data_clean.astype(np.int)
            if all(data_clean == data_int):
                data_clean = data_int
                dtype = "ordinal"

                if self.verbose:
                    print("datatype check: variable was casted as ordinal.")

        # descriptive analysis
        d_n_zero = np.count_nonzero(data_clean == 0)
        d_p_zero = d_n_zero / self._n_samples
        d_n_neg = np.count_nonzero(data_clean < 0)
        d_n_pos = n_informed - d_n_zero - d_n_neg

        d_min = np.min(data_clean)
        d_max = np.max(data_clean)
        d_mean = np.mean(data_clean)
        d_std = np.std(data_clean)
        values, counts = np.unique(data_clean, return_counts=True)
        d_mode = values[np.argmax(counts)] if counts.size else np.nan

        if data_clean.size:
            percentiles = np.percentile(data_clean, [1, 25, 50, 75, 99])
            d_percentile_1 = percentiles[0]
            d_percentile_25 = percentiles[1]
            d_median = percentiles[2]
            d_percentile_75 = percentiles[3]
            d_percentile_99 = percentiles[4]
            iqr = d_percentile_75 - d_percentile_25
            d_outlier_low = d_percentile_25 - 1.5 * iqr
            d_outlier_high = d_percentile_75 + 1.5 * iqr
            d_n_outlier_low = np.count_nonzero(data_clean < d_outlier_low)
            d_n_outlier_high = np.count_nonzero(data_clean > d_outlier_high)

            eps = 1e-15
            d_coeff_disp_param = d_mean / (d_std + eps)
            d_q1pq3 = d_percentile_25 + d_percentile_75 + eps
            d_coeff_disp_nonparam = iqr / d_q1pq3

            if iqr:
                hist = np.histogram(
                    data_clean[(data_clean >= d_outlier_low) & (
                        data_clean <= d_outlier_high)], bins="sturges")
            else:
                hist = np.histogram(
                    data_clean[(data_clean >= d_outlier_low) & (
                        data_clean <= d_outlier_high)])

            d_hist_cols, d_hist_pos = hist

            idx_concentration = np.argmax(d_hist_cols)
            d_concentration = d_hist_cols[idx_concentration] / n_informed
            d_concentration_interval = (
                d_hist_pos[idx_concentration],
                d_hist_pos[idx_concentration + 1])

            # overwrite with HHI
            d_concentration = np.sum((d_hist_cols / n_informed) ** 2)
        else:
            d_percentile_1 = np.nan
            d_percentile_25 = np.nan
            d_median = np.nan
            d_percentile_75 = np.nan
            d_percentile_99 = np.nan
            d_outlier_low = np.nan
            d_outlier_high = np.nan
            d_n_outlier_low = np.nan
            d_n_outlier_high = np.nan
            d_coeff_disp_param = np.nan
            d_coeff_disp_nonparam = np.nan
            d_hist_cols = []
            d_hist_pos = []
            d_concentration = np.nan
            d_concentration_interval = (np.nan, np.nan)

        # variable dict
        fields = [
            "dtype", "n_special", "p_special", "u_special", "n_u_special",
            "n_missing", "p_missing", "n_informed", "d_n_zero", "d_p_zero",
            "d_n_neg", "d_n_pos", "d_min", "d_max", "d_mean",
            "d_std", "d_mode", "d_percentile_1", "d_percentile_25", "d_median",
            "d_percentile_75", "d_percentile_99", "d_outlier_low",
            "d_outlier_high", "d_n_outlier_low", "d_n_outlier_high",
            "d_coeff_disp_param", "d_coeff_disp_nonparam", "d_hist_cols",
            "d_hist_pos", "d_concentration", "d_concentration_interval"]

        info = [
            dtype, n_special, p_special, u_special, n_u_special, n_missing,
            p_missing, n_informed, d_n_zero, d_p_zero, d_n_neg, d_n_pos, d_min,
            d_max, d_mean, d_std, d_mode, d_percentile_1, d_percentile_25,
            d_median, d_percentile_75, d_percentile_99, d_outlier_low,
            d_outlier_high, d_n_outlier_low, d_n_outlier_high,
            d_coeff_disp_param, d_coeff_disp_nonparam, d_hist_cols, d_hist_pos,
            d_concentration, d_concentration_interval]

        dict_variable_info = dict(zip(fields, info))

        # temporal analysis
        if self.date is not None:
            dates_clean = self._dates[idx_clean]
            dict_variable_t_info = _temporal_numerical(
                data.values, data_clean.values, self._dates, dates_clean,
                self._unique_dates, self.special_values)

            dict_variable_info = {**dict_variable_info, **dict_variable_t_info}

        return dict_variable_info

    def _run_variable_categorical(self, data, dtype):
        [idx_special, n_special, p_special, u_special, n_u_special,
            idx_missing, n_missing, p_missing,
            n_informed] = _information_metrics(
                data, self._n_samples, self.special_values)

        # clean data to compute descriptive statistics
        if self.special_values:
            idx_clean = (~idx_special & ~idx_missing)
        else:
            idx_clean = ~idx_missing

        data_clean = data[idx_clean]

        # number of categories
        if data_clean.size:
            unique_categories = data_clean.value_counts()
            u_categories = unique_categories.index.values
            n_u_categories = unique_categories.values
            d_n_categories = len(u_categories)
            d_most_freq_category = u_categories[0]
            d_p_most_freq_category = n_u_categories[0] / n_informed

            # top 10 most frequent categories
            max_categories = min(d_n_categories, 10)
            d_top_categories = u_categories[:max_categories]
            d_n_top_categories = n_u_categories[:max_categories]

            # histogram
            d_hist_cols = d_n_top_categories
            d_hist_pos = d_top_categories

            # caclulate concentration information mesure HHI
            d_hhi = np.sum(np.square(n_u_categories / n_informed))
            q = 1.0 / d_n_categories
            d_hhi_norm = (d_hhi - q) / (1.0 - q)
        else:
            d_n_categories = np.nan
            d_most_freq_category = np.nan
            d_p_most_freq_category = np.nan
            d_top_categories = []
            d_n_top_categories = []
            d_hist_cols = []
            d_hist_pos = []
            d_hhi = np.nan
            d_hhi_norm = np.nan

        # variable dict
        fields = [
            "dtype", "n_special", "p_special", "u_special",
            "n_u_special", "n_missing", "p_missing", "n_informed",
            "d_n_categories", "d_most_freq_category", "d_p_most_freq_category",
            "d_top_categories", "d_n_top_categories", "d_hist_cols",
            "d_hist_pos", "d_hhi", "d_hhi_norm"]

        info = [
            dtype, n_special, p_special, u_special, n_u_special, n_missing,
            p_missing, n_informed, d_n_categories, d_most_freq_category,
            d_p_most_freq_category, d_top_categories, d_n_top_categories,
            d_hist_cols, d_hist_pos, d_hhi, d_hhi_norm]

        dict_variable_info = dict(zip(fields, info))

        # temporal analysis
        if self.date is not None:
            dates_clean = self._dates[idx_clean]
            dict_variable_t_info = _temporal_categorical(
                data.values, data_clean.values, self._dates, dates_clean,
                self._unique_dates, self.special_values, d_top_categories)

            dict_variable_info = {**dict_variable_info, **dict_variable_t_info}

        return dict_variable_info

    def _decision_engine(self):
        """Set action to be taken for each variable."""
        MSG_PERCENTAGE = "% {} values above threshold: {:.2%} > {:.2%}"
        MSG_DIVERGENCE = "{} value above threshold {:.3f} > {:.3f}"

        for column in self._column_names:
            t_min_info = None
            t_min_data = None
            if self._dict_variables[column]["p_missing"] > self.max_p_missing:
                self._dict_variables[column]["status"] = STATUS_MISSING
                msg = MSG_PERCENTAGE.format(
                    "missing", self._dict_variables[column]["p_missing"],
                    self.max_p_missing)
                self._dict_variables[column]["comment"] = msg

            elif (self._dict_variables[column]["p_special"]
                    > self.max_p_special):
                self._dict_variables[column]["status"] = STATUS_SPECIAL
                msg = MSG_PERCENTAGE.format(
                    "special", self._dict_variables[column]["p_special"],
                    self.max_p_special)
                self._dict_variables[column]["comment"] = msg

            elif self.date is not None:
                t_min_info = min(
                    self._dict_variables[column]["t_info_div_uniform"],
                    self._dict_variables[column]["t_info_div_exponential"])

                t_min_data = min(
                    self._dict_variables[column]["t_data_div_uniform"],
                    self._dict_variables[column]["t_data_div_exponential"])

                if t_min_info > self.max_divergence:
                    status = STATUS_INFO_DIVERGENCE
                    msg = MSG_DIVERGENCE.format(
                        "t_info_div", t_min_info, self.max_divergence)
                    self._dict_variables[column]["status"] = status
                    self._dict_variables[column]["comment"] = msg
                elif t_min_data > self.max_divergence:
                    status = STATUS_DATA_DIVERGENCE
                    msg = MSG_DIVERGENCE.format(
                        "t_data_div", t_min_data, self.max_divergence)
                    self._dict_variables[column]["status"] = status
                    self._dict_variables[column]["comment"] = msg
                else:
                    self._dict_variables[column]["status"] = STATUS_OK
            else:
                self._dict_variables[column]["status"] = STATUS_OK

            # set action for each column according to its status
            status = self._dict_variables[column]["status"]
            if status in STATUS_REMOVE:
                self._dict_variables[column]["action"] = "remove"
            elif status in STATUS_REVIEW:
                self._dict_variables[column]["action"] = "review"
            else:
                self._dict_variables[column]["action"] = "keep"

            action = self._dict_variables[column]["action"]
            self._dict_variables[column]["recommended_action"] = action

            # compute quality score
            p_missing = self._dict_variables[column]["p_missing"]
            p_special = self._dict_variables[column]["p_special"]
            if self.date is None:
                t_min_info = None
                t_min_data = None

            self._dict_variables[column]["quality_score"] = _quality_score(
                p_missing, p_special, t_min_info, t_min_data)

    def _transform_remove(self, data, mode):
        """Remove unnecessary variables in-place."""
        AGGR_COMMENT = "auto: remove due to aggressive mode"
        BASIC_COMMENT = "auto: keep due to basic mode"
        DATE_COMMENT = "auto: keep date column"
        TARGET_COMMENT = "auto: keep target column"

        remove = [column for column in self._column_names if
                  self._dict_variables[column]["action"] == "remove"]

        review = [column for column in self._column_names if
                  self._dict_variables[column]["action"] == "review"]

        if mode == "aggressive":
            if self.verbose:
                print("columns with action='remove' or action='review' "
                      "except the date and target column if supplied are "
                      "dropped.")

            for column in review:
                if self.date == column:
                    self._dict_variables[column]["action"] = "keep"
                    self._dict_variables[column]["auto_comment"] = DATE_COMMENT
                elif self.target == column:
                    self._dict_variables[column]["action"] = "keep"
                    msg = TARGET_COMMENT
                    self._dict_variables[column]["auto_comment"] = msg
                else:
                    remove.append(column)
                    self._dict_variables[column]["action"] = "remove"
                    self._dict_variables[column]["auto_comment"] = AGGR_COMMENT

        elif mode == "basic":
            if self.verbose:
                print("only columns with action='remove' are dropped.")

            # if recommended action is not removed then keep
            for column in review:
                if self.date == column:
                    msg = DATE_COMMENT
                    self._dict_variables[column]["auto_comment"] = msg
                elif self.target == column:
                    msg = TARGET_COMMENT
                    self._dict_variables[column]["auto_comment"] = msg
                else:
                    msg = BASIC_COMMENT
                    self._dict_variables[column]["auto_comment"] = msg

                self._dict_variables[column]["action"] = "keep"

        self._n_vars_remove = len(remove)

        # drop column in-place
        data.drop(remove, axis=1, inplace=True)

        # garbage collector
        gc.collect()

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
            raise NotRunException(self, "run")

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
            raise NotRunException(self, "run")

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
            raise NotRunException(self, "run")

        if variable not in self._dict_variables.keys():
            raise ValueError("variable {} not in data.".format(variable))

        if action not in ("keep", "remove"):
            raise ValueError("action {} not supported.".format(action))

        self._dict_variables[variable]["user_action"] = action
        self._dict_variables[variable]["user_comment"] = comment

        self._dict_variables[variable]["action"] = action
