"""
Exploratory Data Analysis (EDA) - univariate analysis with Spark
"""

# Authors: Fernando Gallego Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2019.

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql.functions as F

from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StringType

from ...core.exceptions import NotRunException
from ...data_analysis.univariate import Univariate
from ...data_analysis.util import js_divergence_multivariate
from ...reporting.util import reporting_output_format
from .spark_util import calculate_all_static_metrics, calculate_cat_metrics
from .spark_util import check_dtype, histogram_rdd, histogram2_rdd
from .spark_util import time_groups_cat_rdd, time_metrics_cat_rdd
from .spark_util import time_metrics_rdd, check_date_format


AGGR_COMMENT = "auto: remove due to aggressive mode"
BASIC_COMMENT = "auto: keep due to basic mode"
DATE_COMMENT = "auto: keep date column"
TARGET_COMMENT = "auto: keep target column"


def _univariate_results(dict_variables, step, date, format):
    """Return run or transform step results as a dataframe or json."""

    # common metrics
    common_cols = [
        "name", "dtype", "recommended_action", "status", "comment",
        "n_missing", "p_missing", "n_special", "p_special", "quality_score"]

    # numerical/ordinal metrics
    num_cols = [
        "d_n_zero", "d_p_zero", "d_n_neg", "d_n_pos", "d_min", "d_max",
        "d_mean", "d_std", "d_percentile_1", "d_percentile_25",
        "d_median", "d_percentile_75", "d_percentile_99", "d_outlier_low",
        "d_outlier_high", "d_n_outlier_low", "d_n_outlier_high",
        "d_coeff_disp_param", "d_coeff_disp_nonparam", "d_hist_cols",
        "d_hist_pos", "d_concentration", "d_concentration_interval"]

    # categorical/nominal metrics
    cat_cols = [
        "d_n_categories", "d_most_freq_category",
        "d_p_most_freq_category", "d_top_categories", "d_n_top_categories",
        "d_hist_cols", "d_hist_pos", "d_hhi", "d_hhi_norm"]

    # transform
    order_cols_transform = [
        "name", "dtype", "action", "status", "comment",
        "recommended_action", "user_action", "user_comment", "auto_comment"]

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

        num_results = results[results.dtype.isin(["numerical", "ordinal"])]
        if num_results.shape[0] > 0:
            num_results = num_results[num_order_cols]
        else:
            num_results = pd.DataFrame()

        cat_results = results[results.dtype.isin(["categorical", "nominal"])]
        if cat_results.shape[0] > 0:
            cat_results = cat_results[cat_order_cols]
        else:
            cat_results = pd.DataFrame()

        return (reporting_output_format(num_results, format),
                reporting_output_format(cat_results, format))
    else:
        for col in ["comment", "user_action", "user_comment", "auto_comment"]:
            if col not in results.columns:
                results[col] = np.nan

        results = results[order_cols_transform]
        return reporting_output_format(results, format)


class UnivariateSpark(Univariate):
    """
    Univariate data analysis with Spark.

    Perform basic exploratory data analysis on a dataset.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        HDFS dataframe.

    spark_context : pyspark.sql.session.SparkSession
        Entry point to the services of Spark.

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

    percentile_acc : int (default=1000)
        The parameter controls approximation accuracy at the cost of memory.
        Higher values yield better approximations, and the default is 1000.
        When the number of distinct values is smaller than this value, it
        gives an exact percentile value.

    verbose : int or boolean (default=False)
        Controls verbosity of output.
    """
    def __init__(self, df, spark_context, target=None, date=None,
                 variables_nominal=None, special_values=None,
                 max_p_missing=0.99, max_p_special=0.99, max_divergence=0.2,
                 percentile_acc=1000, verbose=False):
        # inputs
        self.df = df
        self.spark_context = spark_context
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

        self.max_p_missing = max_p_missing
        self.max_p_special = max_p_special
        self.max_divergence = max_divergence
        self.percentile_acc = percentile_acc
        self.verbose = verbose

        # database and run results
        self._column_names = []
        self._col_types = None
        self._n_samples = None
        self._n_vars = None
        self._n_vars_remove = None
        self._dict_variables = {}
        self._target_dtype = None
        self._date = None
        self._unique_dates = []

        # time variables
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
        Returns information and flags for each variable.

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

    def run(self):
        """
        Runs univariate analysis in pyspark.

        Note that temporal analysis is only performed if a date variable is
        provided during instantiation.
        """

        if not isinstance(self.df, DataFrame):
            raise TypeError("df must be pyspark.sql.DataFrame.")

        if not isinstance(self.spark_context, SparkSession):
            raise TypeError("spark_context must be "
                            "pyspark.sql.session.SparkSession.")

        self._column_names = list(self.df.columns)
        self._n_vars = len(self._column_names)
        self._n_samples = self.df.count()

        # check whether given target and date are in dataframe
        if self.target is not None and self.target not in self._column_names:
            raise ValueError("target variable {} not available in dataframe."
                             .format(self.target))

        if self.date is not None and self.date not in self._column_names:
            raise ValueError("date variable {} not available in dataframe."
                             .format(self.date))

        # keep date data for temporal analysis
        if self.date is not None:
            if self.verbose:
                print("Checking date format.")
            date_check_format = check_date_format(self.df, self.date)
            if date_check_format[0][self.date] > 0:
                raise ValueError("date variable has incorrect format. "
                                 "Correct format: integer yyyymm.")
            if self.verbose:
                print("Format correct!")
            self._column_names.remove(self.date)
            self._date = self.df.select([self.date])

            if self.df.select(self.date).dtypes[0][1] != "string":
                self.df = self.df.withColumn(
                    self.date, self.df[self.date].cast(StringType()))

        # target information
        if self.target is not None:
            self._column_names.remove(self.target)

        # run univariate
        self._run()

        self._is_run = True

    def transform(self, mode="basic"):
        """
        Transforms input raw dataset in-place.

        Reduce the raw dataset by removing columns with action flag equal to
        remove.

        Parameters
        ----------
        mode : str
            Transformation mode, options are "aggressive" and "basic". If
            ``mode=="aggressive"`` columns tagged with action "remove" and
            with status "id" and "date" (except the date column supplied) are
            dropped. If ``mode=="basic"`` only columns tagged as "remove" are
            dropped.
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        if mode not in ("aggressive", "basic"):
            raise ValueError("mode {} is not supported.".format(mode))

        self._transform_mode = mode

        time_init = time.perf_counter()
        self._transform_remove(mode=mode)

        self._is_transformed = True
        self._time_transform = time.perf_counter() - time_init

    def variable_stats(self, variable):
        """Prints and plots the main info of the specified variable.

        Parameters
        ----------
        variable : str
            Name of the variable.
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        if variable not in self._column_names:
            raise ValueError("Variable {} is not present in the "
                             "db.".format(variable))

        print("missing:", self._dict_variables[variable]["n_missing"])
        print("special:", self._dict_variables[variable]["n_special"])
        print("informed:", self._dict_variables[variable]["n_informed"])
        print("-----")

        if self._dict_variables[variable]["dtype"] in ["numerical", "ordinal"]:
            self._variable_stats_num(variable)
        elif (self._dict_variables[variable]["dtype"] in
                ["nominal", "categorical"]):
            self._variable_stats_cat(variable)

    def _run(self):
        """Executes the calculations for all types of variables"""
        time_init = time.perf_counter()

        self._col_types = {"numerical": [], "ordinal": [],
                           "categorical": [], "nominal": [], "other": []}
        for tup in self.df.dtypes:
            if tup[0] not in [self.date, self.target]:
                self._col_types[check_dtype(
                    tup[0], tup[1], self.variables_nominal)].append(tup[0])
        # numerical and ordinal
        vars_numerical = (self._col_types["ordinal"] +
                          self._col_types["numerical"])
        if len(vars_numerical) > 0:
            self._run_variable_numerical(vars_numerical)
        # categorical and nominal
        vars_categorical = (self._col_types["nominal"] +
                            self._col_types["categorical"] +
                            self._col_types["other"])
        if len(vars_categorical) > 0:
            self._run_variable_categorical(vars_categorical)

        for key_type in self._col_types:
            for var in self._col_types[key_type]:
                self._dict_variables[var]["dtype"] = key_type

        # set actions
        self._decision_engine()

        self._time_run = time.perf_counter() - time_init

    def _run_variable_categorical(self, variables):
        """Calculations for categorical variables"""
        time_init_cat = time.perf_counter()

        if self.verbose:
            print("categorical metrics")
        dict_var_cat = calculate_cat_metrics(
            self.df, variables, self.special_values, self.verbose)
        time_metrics = (time.perf_counter() - time_init_cat)
        if self.verbose:
            print(("metrics cat. - time (s): {:>10.2f}s\n----").format(
                time_metrics))
        if self.date is not None:
            if self.verbose:
                print("time info")
            time_cat = time_metrics_cat_rdd(
                self.df, variables + [self.date], self.special_values,
                self.date)
            time_cat = sorted(time_cat, key=lambda x: x[0])
            self._unique_dates = [tup[0] for tup in time_cat]
            for i, var in enumerate(variables):
                dict_var_cat[var]["dates"] = [
                        tup[0] for tup in time_cat]
                dict_var_cat[var]["t_n_missing"] = [
                    tup[1][0][i] for tup in time_cat]
                dict_var_cat[var]["t_n_special"] = [
                    tup[1][1][i] for tup in time_cat]
                dict_var_cat[var]["t_n_info"] = [
                    tup[1][2][i] for tup in time_cat]
            time_time_info = (time.perf_counter() - time_init_cat -
                              time_metrics)
            if self.verbose:
                print(("Time information (s): {:>10.2f}s\n----").format(
                    time_time_info))

            if self.verbose:
                print("time groups")
            time_group_cat = time_groups_cat_rdd(
                self.df, variables, self.special_values,
                self.date, dict_var_cat)

            for var in variables:
                dict_var_cat[var]["cat_dates"] = {elt: {
                    str(tpl[0]): 0 for tpl in dict_var_cat[var][
                        "10_categories"]
                    } for elt in dict_var_cat[var]["dates"]}

            for key in time_group_cat:
                split_key = key.split("$___$")
                dict_var_cat[split_key[1]]["cat_dates"][
                    split_key[2]][split_key[0]] = time_group_cat[key]

            time_time_groups = (time.perf_counter() - time_init_cat -
                                time_metrics - time_time_info)
            if self.verbose:
                print(("Time groups (s): {:>10.2f}s\n----").format(
                    time_time_groups))

            for var in variables:
                missings = dict_var_cat[var]["t_n_missing"]
                informed = dict_var_cat[var]["t_n_info"]
                specials = dict_var_cat[var]["t_n_special"]

                dict_var_cat[var]["t_missing"] = [
                    elt/(specials[i]+elt+informed[i])
                    for i, elt in enumerate(missings)]
                dict_var_cat[var]["t_special"] = [
                    elt/(missings[i]+elt+informed[i])
                    for i, elt in enumerate(specials)]
                dict_var_cat[var]["t_info"] = [
                    elt/(specials[i]+elt+missings[i])
                    for i, elt in enumerate(informed)]

                dict_var_cat[var]["t_n_months_no_info"] = len(
                    [1 for elt in dict_var_cat[var]["t_n_info"]
                     if elt == 0])

                # calculate percentages of categories ocupation for each date
                # this is needed for reporting.
                dict_var_cat[var]["t_rest"] = [1.] * len(self._unique_dates)
                for i, val in enumerate(dict_var_cat[var]['d_top_categories']):
                    vect = []
                    for j, date in enumerate(self._unique_dates):
                        if dict_var_cat[var]['t_n_info'][j] > 0:
                            perc_val = (
                                dict_var_cat[var]["cat_dates"][date][val] /
                                dict_var_cat[var]['t_n_info'][j])
                            vect.append(perc_val)
                            dict_var_cat[var]["t_rest"][j] -= vect[-1]
                        else:
                            vect.append(0)
                            dict_var_cat[var]["t_rest"][j] = 0.
                    dict_var_cat[var]["t_c" + str(i)] = vect

                # compute divergence measures
                X_info = np.asarray([dict_var_cat[var]["t_info"],
                                     dict_var_cat[var]["t_special"],
                                     dict_var_cat[var]["t_missing"]])
                dict_var_cat[var][
                    "t_info_div_uniform"
                    ] = js_divergence_multivariate(X_info, "uniform")
                dict_var_cat[var][
                    "t_info_div_exponential"
                    ] = js_divergence_multivariate(X_info, "exponential")

                X_data = []
                suma_date = {elt: sum(
                    dict_var_cat[var]["cat_dates"][elt].values()
                    ) for elt in dict_var_cat[var]["dates"]}
                for vec in dict_var_cat[var]["10_categories"]:
                    vec_tuple = []
                    for key in dict_var_cat[var]["cat_dates"]:
                        if suma_date[key] > 0:
                            vec_tuple.append((
                                key,
                                (dict_var_cat[var]["cat_dates"][key][vec[0]] /
                                 suma_date[key])))
                        else:
                            vec_tuple.append((key, 0))
                    sorted_tuple = sorted(vec_tuple, key=lambda x: x[0])
                    X_data.append([elt[1] for elt in sorted_tuple])

                X_data = np.asarray(X_data)
                dict_var_cat[var][
                    "t_data_div_uniform"
                    ] = js_divergence_multivariate(X_data, "uniform")
                dict_var_cat[var][
                    "t_data_div_exponential"
                    ] = js_divergence_multivariate(X_data, "exponential")

        self._time_run_categorical = time.perf_counter() - time_init_cat
        self._time_run_nominal = 0

        self._dict_variables = {**self._dict_variables, **dict_var_cat}

    def _run_variable_numerical(self, variables):
        """Calculations for numerical variables"""
        time_init_num = time.perf_counter()

        # Metrics and percentiles
        if self.verbose:
            print("metrics and percentiles")
        self._dict_variables = calculate_all_static_metrics(
            self.df, variables, self.special_values,
            self.spark_context, error_P=1000, verbose=self.verbose)
        time_metrics = (time.perf_counter() - time_init_num)
        if self.verbose:
            print(("metrics and Perc. - time (s): {:>10.2f}s\n----").format(
                time_metrics))

        # Histogram
        if self.verbose:
            print("histogram part 1")
        histogram_rdd(self.df, variables, self._dict_variables,
                      self.special_values)
        print("histogram part 2")
        histogram2_rdd(self.df, variables, self._dict_variables,
                       self.special_values)
        time_hist = (time.perf_counter() - time_init_num - time_metrics)
        if self.verbose:
            print(("histogram - time (s): {:>10.2f}s\n----").format(
                time_hist))

        # Time information
        if self.date is not None:
            if self.verbose:
                print("Time Information")
            time_var = time_metrics_rdd(self.df, variables + [self.date],
                                        self.special_values, self.date)
            time_var = sorted(time_var, key=lambda x: x[0])
            self._unique_dates = [tup[0] for tup in time_var]
            for i, var in enumerate(variables):
                self._dict_variables[var]["dates"] = [
                        tup[0] for tup in time_var]
                self._dict_variables[var]["t_n_missing"] = [
                    tup[1][0][i] for tup in time_var]
                self._dict_variables[var]["t_n_special"] = [
                    tup[1][1][i] for tup in time_var]
                self._dict_variables[var]["t_n_info"] = [
                    tup[1][2][i] for tup in time_var]
            time_time_info = (time.perf_counter() - time_init_num -
                              time_metrics - time_hist)
            if self.verbose:
                print(("Time information (s): {:>10.2f}s\n----").format(
                    time_time_info))

            # Time percentiles
            if self.verbose:
                print("Time Percentiles")
            new_df = self.df.select([
                F.when(~(F.col(c).isin(self.special_values)),
                       F.col(c)).otherwise(
                    None).alias(c) for c in variables + [self.date]])
            new_df.registerTempTable("df_quantiles2")

            string_sql_hive = "select " + self.date
            array_perc = "ARRAY(0.25,0.5,0.75)"
            for col_name in variables:
                string_sql_hive += (", percentile_approx(" + col_name + "," +
                                    array_perc + ",1000) as " + col_name)
            string_sql_hive += (" from " + "df_quantiles2" +
                                " group by " + self.date)

            time_perc = self.spark_context.sql(string_sql_hive).collect()
            time_perc = sorted(time_perc, key=lambda x: x[self.date])
            for key in variables:
                self._dict_variables[key]["t_p25"] = []
                self._dict_variables[key]["t_median"] = []
                self._dict_variables[key]["t_p75"] = []
                for row in time_perc:
                    if row[key] is not None:
                        self._dict_variables[key]["t_p25"].append(row[key][0])
                        self._dict_variables[key]["t_median"].append(
                            row[key][1])
                        self._dict_variables[key]["t_p75"].append(row[key][2])
                    else:
                        self._dict_variables[key]["t_p25"].append(np.nan)
                        self._dict_variables[key]["t_median"].append(np.nan)
                        self._dict_variables[key]["t_p75"].append(np.nan)

            time_time_perc = (time.perf_counter() - time_init_num -
                              time_metrics - time_hist - time_time_info)

            for var in variables:
                missings = self._dict_variables[var]["t_n_missing"]
                informed = self._dict_variables[var]["t_n_info"]
                specials = self._dict_variables[var]["t_n_special"]

                self._dict_variables[var]["t_missing"] = [
                    elt/(specials[i]+elt+informed[i])
                    for i, elt in enumerate(missings)]
                self._dict_variables[var]["t_special"] = [
                    elt/(missings[i]+elt+informed[i])
                    for i, elt in enumerate(specials)]
                self._dict_variables[var]["t_info"] = [
                    elt/(specials[i]+elt+missings[i])
                    for i, elt in enumerate(informed)]

                self._dict_variables[var]["t_n_months_no_info"] = len(
                    [1 for elt in self._dict_variables[var]["t_n_info"]
                     if elt == 0])

                # compute divergence measures
                X_info = np.asarray([self._dict_variables[var]["t_info"],
                                     self._dict_variables[var]["t_special"],
                                     self._dict_variables[var]["t_missing"]])
                self._dict_variables[var][
                    "t_info_div_uniform"
                    ] = js_divergence_multivariate(X_info, "uniform")
                self._dict_variables[var][
                    "t_info_div_exponential"
                    ] = js_divergence_multivariate(X_info, "exponential")

                prob_p25 = []
                prob_p50 = []
                prob_p75 = []
                for i, elt in enumerate(self._dict_variables[var]["t_n_info"]):
                    if elt == 0:
                        prob_p25.append(0)
                        prob_p50.append(0)
                        prob_p75.append(0)
                    else:
                        prob_p25.append(
                            self._dict_variables[key]["t_p25"][i]/elt)
                        prob_p50.append(
                            self._dict_variables[key]["t_median"][i]/elt)
                        prob_p75.append(
                            self._dict_variables[key]["t_p75"][i]/elt)
                # X_data = np.asarray([prob_p25, prob_p50, prob_p75])
                self._dict_variables[var][
                    "t_data_div_uniform"
                    ] = 0  # js_divergence_multivariate(X_data, "uniform")
                self._dict_variables[var][
                    "t_data_div_exponential"
                    ] = 0  # js_divergence_multivariate(X_data, "exponential")

            if self.verbose:
                print(("Time percentiles (s): {:>10.2f}s\n----").format(
                    time_time_perc))

        self._time_run_numerical = time.perf_counter() - time_init_num
        self._time_run_ordinal = 0

    def _transform_remove(self, mode):
        """Removes unnecessary variables in-place."""

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
                    self._dict_variables[column][
                        "auto_comment"] = DATE_COMMENT
                elif self.target == column:
                    self._dict_variables[column]["action"] = "keep"
                    self._dict_variables[column][
                        "auto_comment"] = TARGET_COMMENT
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
                    self._dict_variables[column]["auto_comment"] = DATE_COMMENT
                elif self.target == column:
                    self._dict_variables[column][
                        "auto_comment"] = TARGET_COMMENT
                else:
                    self._dict_variables[column][
                        "auto_comment"] = BASIC_COMMENT

                self._dict_variables[column]["action"] = "keep"

        self._n_vars_remove = len(remove)

        # drop column in-place
        self.df = self.df.drop(*remove)

    def _variable_stats_cat(self, var):
        """Prints and plots the main information of categorical vars."""
        print("categories:", self._dict_variables[var]["d_n_categories"])
        # groups
        plt.bar([elt[0] for elt in self._dict_variables[var]["10_categories"]],
                [elt[1] for elt in self._dict_variables[var]["10_categories"]])
        plt.bar(["Others"], [self._dict_variables[var]["others_categories"]])
        plt.xticks(rotation='vertical')
        plt.title(var)
        plt.show()
        plt.close()
        # time information
        if self.date is not None:
            dates = self._dict_variables[var]["dates"]
            plt.plot(dates,
                     self._dict_variables[var]["t_missing"], label="missings")
            plt.plot(dates,
                     self._dict_variables[var]["t_info"], label="informed")
            plt.plot(dates,
                     self._dict_variables[var]["t_special"], label="specials")
            plt.legend()
            plt.xlabel("dates")
            plt.ylabel("%")
            plt.title(var)
            plt.show()
            plt.close()
            # time groups
            suma_date = {elt: sum(
                self._dict_variables[var]["cat_dates"][elt].values()
                ) for elt in dates}
            for vec in self._dict_variables[var]["10_categories"]:
                vec_tuple = [(
                    key, self._dict_variables[var]["cat_dates"][key][vec[0]] /
                    suma_date[key]
                    ) for key in self._dict_variables[var]["cat_dates"]]
                sorted_tuple = sorted(vec_tuple, key=lambda x: x[0])
                plt.plot([elt[0] for elt in sorted_tuple],
                         [elt[1] for elt in sorted_tuple], label=vec[0])
            plt.legend()
            plt.xlabel("dates")
            plt.ylabel("%")
            plt.show()
            plt.close()

    def _variable_stats_num(self, var):
        """Prints and plots the main information of numerical vars."""
        print("negative:", self._dict_variables[var]["d_n_neg"])
        print("zeros:", self._dict_variables[var]["d_n_zero"])
        print("positive:", self._dict_variables[var]["d_n_pos"])
        print("-----")
        print("min:", self._dict_variables[var]["d_min"])
        print("P1:", self._dict_variables[var]["d_percentile_1"])
        print("P25:", self._dict_variables[var]["d_percentile_25"])
        print("P50:", self._dict_variables[var]["d_median"])
        print("mean:", self._dict_variables[var]["d_mean"])
        print("P75:", self._dict_variables[var]["d_percentile_75"])
        print("P99:", self._dict_variables[var]["d_percentile_99"])
        print("max:", self._dict_variables[var]["d_max"])
        # histogram
        vect_hist_plot = [self._dict_variables[var]["d_hist_cols"][i]
                          for i in range(self._dict_variables[var]["n_bins"])]
        if self._dict_variables[var]["increment"] == 0:
            bar_width = 0.8
        else:
            bar_width = (
                (self._dict_variables[var]["d_outlier_high"] -
                 self._dict_variables[var]["d_outlier_low"]) /
                (self._dict_variables[var]["n_bins"] + 2))
        plt.bar(self._dict_variables[var]["d_hist_pos"][:-1],
                vect_hist_plot, width=bar_width)
        plt.bar([self._dict_variables[var]["d_outlier_high"]+1],
                self._dict_variables[var]["d_n_outlier_high"],
                width=bar_width)
        plt.bar([self._dict_variables[var]["d_outlier_low"]-1],
                self._dict_variables[var]["d_n_outlier_low"],
                width=bar_width)
        plt.show()
        plt.close()
        # time information
        if self.date is not None:
            dates = self._dict_variables[var]["dates"]
            plt.plot(dates,
                     self._dict_variables[var]["t_missing"], label="missings")
            plt.plot(dates,
                     self._dict_variables[var]["t_info"], label="informed")
            plt.plot(dates,
                     self._dict_variables[var]["t_special"], label="specials")
            plt.legend()
            plt.xlabel("dates")
            plt.ylabel("%")
            plt.title(var)
            plt.show()
            plt.close()
            # time percentiles
            plt.plot(dates, self._dict_variables[var]["t_p25"], label="P25")
            plt.plot(dates, self._dict_variables[var]["t_median"], label="P50")
            plt.plot(dates, self._dict_variables[var]["t_p75"], label="P75")
            plt.legend()
            plt.xlabel("dates")
            plt.ylabel("%")
            plt.title(var)
            plt.show()
            plt.close()
