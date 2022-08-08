"""
Univariate utils for spark univariate.
"""

# Authors: Fernando Gallego Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2019.

import re

import numpy as np
import pyspark.sql.functions as F

from pyspark.sql.types import IntegerType


MAX_CATEGORIES = 10


def calculate_all_static_metrics(
        df_spark, variables, special_values, spark_context, error_P=1000,
        verbose=False):
    """
    Calculates the main metrics, percentiles and histogram for numerical
    variables.
    """
    dict_info = {
        "dtype": None, "status": None, "comment": None, "action": None,
        "recommended_action": None, "quality_score": 0,
        "p_missing": 0, "p_special": 0, "n_missing": 0, "n_special": 0,
        "u_special": [],
        "n_informed": 0, "d_n_zero": 0, "d_p_zero": 0, "d_n_neg": 0,
        "d_n_pos": 0, "d_min": None, "d_max": None, "d_mean": None,
        "d_std": None, "d_percentile_1": None, "d_percentile_25": None,
        "d_median": None, "d_percentile_75": None, "d_percentile_99": None,
        "d_mode": None,
        "iqr": None, "d_outlier_low": None, "d_n_outlier_low": None,
        "d_outlier_high": None, "d_n_outlier_high": None,
        "d_coeff_disp_param": None, "d_coeff_disp_nonparam": None,
        "n_bins": 0, "increment": 0, "d_hist_pos": [], "d_hist_cols": [],
        "d_concentration": 0, "d_concentration_interval": 0,
        "t_p25": [], "t_median": [], "t_p75": [], "t_n_months_no_info": 0,
        "t_missing": [], "t_special": [], "t_info": [], "t_mean": [],
        "t_info_div_uniform": 0, "t_info_div_exponential": 0,
        "t_data_div_uniform": 0, "t_data_div_exponential": 0
    }
    dict_results = {}

    parametric_metrics = metrics_numerical_rdd(
        df_spark, variables, special_values)
    if verbose:
        print("parametric metrics obtained")

    df_hive = df_spark.select([F.when(
        ~(F.col(c).isin(special_values)), F.col(c)
        ).otherwise(None).alias(c) for c in variables])
    df_hive.registerTempTable("df_first_percentiles")

    string_sql_hive = "select "
    flag_first = True
    array_perc = "ARRAY(0.01,0.25,0.5,0.75,0.99)"
    for col_name in variables:
        if flag_first:
            string_sql_hive += ("percentile_approx(" + col_name + "," +
                                array_perc + "," + str(error_P) + ") as " +
                                col_name)
            flag_first = False
        else:
            string_sql_hive += (", percentile_approx(" + col_name + "," +
                                array_perc + "," + str(error_P) + ") as " +
                                col_name)
    string_sql_hive += " from " + "df_first_percentiles"

    percentiles = spark_context.sql(string_sql_hive).collect()

    if verbose:
        print("percentiles obtained")

    for i, var in enumerate(variables):
        dict_results[var] = dict_info.copy()
        dict_results[var]["n_missing"] = parametric_metrics[0][i]
        dict_results[var]["n_special"] = parametric_metrics[1][i]
        dict_results[var]["n_informed"] = parametric_metrics[4][i]
        dict_results[var]["d_n_zero"] = parametric_metrics[2][i]
        dict_results[var]["d_n_neg"] = parametric_metrics[3][i]
        dict_results[var]["d_n_pos"] = (dict_results[var]["n_informed"] -
                                        dict_results[var]["d_n_zero"] -
                                        dict_results[var]["d_n_neg"])
        dict_results[var]["d_min"] = parametric_metrics[7][i]
        dict_results[var]["d_max"] = parametric_metrics[8][i]
        dict_results[var]["d_mean"] = parametric_metrics[5][i]
        dict_results[var]["d_std"] = parametric_metrics[9][i]

        n_samples = (dict_results[var]["n_special"] +
                     dict_results[var]["n_missing"] +
                     dict_results[var]["n_informed"])
        dict_results[var]["p_missing"] = (
            dict_results[var]["n_missing"] / n_samples)
        dict_results[var]["p_special"] = (
            dict_results[var]["n_special"] / n_samples)
        dict_results[var]["d_p_zero"] = (
            dict_results[var]["d_n_zero"] / n_samples)

        if ((percentiles[0][var] is not None) and
                (len(percentiles[0][var]) == 5)):
            dict_results[var]["d_percentile_1"] = percentiles[0][var][0]
            dict_results[var]["d_percentile_25"] = percentiles[0][var][1]
            dict_results[var]["d_median"] = percentiles[0][var][2]
            dict_results[var]["d_percentile_75"] = percentiles[0][var][3]
            dict_results[var]["d_percentile_99"] = percentiles[0][var][4]
            dict_results[var]["iqr"] = (dict_results[var]["d_percentile_75"] -
                                        dict_results[var]["d_percentile_25"])
            dict_results[var]["d_outlier_low"] = (
                dict_results[var]["d_percentile_25"] -
                1.5 * dict_results[var]["iqr"])
            dict_results[var]["d_outlier_high"] = (
                dict_results[var]["d_percentile_75"] +
                1.5 * dict_results[var]["iqr"])
            dict_results[var]["d_coeff_disp_param"] = (
                dict_results[var]["d_mean"] /
                (dict_results[var]["d_std"] + 1e-15))
            dict_results[var]["d_coeff_disp_nonparam"] = (
                dict_results[var]["iqr"] /
                (dict_results[var]["d_percentile_25"] +
                 dict_results[var]["d_percentile_75"] + 1e-15))

    return dict_results


def calculate_cat_metrics(df_spark, variables, special_values, verbose=False):
    """
    Calculates the main metrics, and categories population for categorical
    variables.
    """
    dict_info = {
        "status": None, "comment": None, "action": None,
        "recommended_action": None, "quality_score": 0, "p_missing": 0,
        "p_special": 0, "n_missing": 0, "n_special": 0, "u_special": [],
        "n_informed": 0, "d_n_categories": 0, "categories": None,
        "10_categories": None, "others_categories": None, "cat_dates": None,
        "d_most_freq_category": None, "d_p_most_freq_category": 0,
        "d_top_categories": [], "d_n_top_categories": [],
        "d_hist_cols": [], "d_hist_pos": [],
        "d_hhi": 0, "d_hhi_norm": 0, "t_info_div_uniform": 0,
        "t_info_div_exponential": 0, "t_data_div_uniform": 0,
        "t_data_div_exponential": 0
    }
    dict_categorias = {}
    dict_results = {}

    fields_t_max_cat = ["t_c{}".format(i) for i in range(MAX_CATEGORIES)]
    fields_t_max_cat.append("t_rest")
    fields_t_cat = [[] for _ in range(MAX_CATEGORIES)] + [[]]
    dict_t_max_cat = dict(zip(fields_t_max_cat, fields_t_cat))

    dict_info = {**dict_info, **dict_t_max_cat}

    parametric_metrics = metrics_cat_rdd(
        df_spark, variables, special_values)

    if verbose:
        print("parametric metrics obtained")

    groups_vars = groups_cat_rdd(df_spark, variables, special_values)
    if verbose:
        print("groups obtained")

    for i, var in enumerate(variables):
        dict_results[var] = dict_info.copy()
        dict_results[var]["categories"] = dict_categorias.copy()
        dict_results[var]["cat_dates"] = dict_categorias.copy()
        dict_results[var]["n_missing"] = parametric_metrics[0][i]
        dict_results[var]["n_special"] = parametric_metrics[1][i]
        dict_results[var]["n_informed"] = parametric_metrics[2][i]

        dict_results[var]["p_missing"] = (
            dict_results[var]["n_missing"] /
            (dict_results[var]["n_special"] + dict_results[var]["n_missing"] +
             dict_results[var]["n_informed"]))
        dict_results[var]["p_special"] = (
            dict_results[var]["n_special"] /
            (dict_results[var]["n_special"] + dict_results[var]["n_missing"] +
             dict_results[var]["n_informed"]))

    for key in groups_vars:
        split_key = key.split("$___$")
        dict_results[split_key[1]][
            "categories"][split_key[0]] = groups_vars[key]

    for elt in variables:
        dict_results[elt]["d_n_categories"] = len(
            dict_results[elt]["categories"])
        dict_results[elt]["10_categories"] = sorted(
            zip(dict_results[elt]["categories"].keys(),
                dict_results[elt]["categories"].values()),
            key=lambda x: x[1], reverse=True)[:MAX_CATEGORIES]
        dict_results[elt]["others_categories"] = (
            dict_results[elt]["n_informed"] - sum(
                elt[1] for elt in dict_results[elt]["10_categories"]))

        if dict_results[elt]["d_n_categories"] > 1:
            dict_results[elt]["d_most_freq_category"] = dict_results[elt][
                "10_categories"][0][0]
            dict_results[elt]["d_p_most_freq_category"] = dict_results[elt][
                "10_categories"][0][1] / dict_results[elt]["n_informed"]
            dict_results[elt]["d_top_categories"] = [
                elt[0] for elt in dict_results[elt]["10_categories"]]
            dict_results[elt]["d_n_top_categories"] = [
                elt[1] for elt in dict_results[elt]["10_categories"]]
            # HHI
            dict_results[elt]["d_hhi"] = np.sum(np.square(np.asarray(
                list(dict_results[elt]["categories"].values())) /
                dict_results[elt]["n_informed"]))
            q = 1.0 / dict_results[elt]["d_n_categories"]
            dict_results[elt]["d_hhi_norm"] = (
                (dict_results[elt]["d_hhi"] - q) / (1.0 - q))
        elif dict_results[elt]["d_n_categories"] == 1:
            dict_results[elt]["d_most_freq_category"] = dict_results[elt][
                "10_categories"][0][0]
            dict_results[elt]["d_p_most_freq_category"] = dict_results[elt][
                "10_categories"][0][1] / dict_results[elt]["n_informed"]
            dict_results[elt]["d_top_categories"] = [
                elt[0] for elt in dict_results[elt]["10_categories"]]
            dict_results[elt]["d_n_top_categories"] = [
                elt[1] for elt in dict_results[elt]["10_categories"]]
            # HHI
            dict_results[elt]["d_hhi"] = 1
            dict_results[elt]["d_hhi_norm"] = 1
        else:
            dict_results[elt]["d_most_freq_category"] = None
            dict_results[elt]["d_p_most_freq_category"] = 0
            dict_results[elt]["d_top_categories"] = []
            dict_results[elt]["d_n_top_categories"] = []
            # HHI
            dict_results[elt]["d_hhi"] = 0
            dict_results[elt]["d_hhi_norm"] = 0
        # histogram equivalent for categorical variables
        dict_results[elt]["d_hist_cols"] = dict_results[elt][
            "d_n_top_categories"]
        dict_results[elt]["d_hist_pos"] = dict_results[elt]["d_top_categories"]
    return dict_results


def check_date_format(df_spark, date_col):
    """Check whether date has the correct format."""

    def is_date_object(s):
        """Check whether object data has the correct date format."""

        # regex: cases yyyymm
        regex = re.compile("[12]\d{3}(0[1-9]|1[0-2])")
        # vectorized implementation of regex for numpy
        vectorize_regex = np.vectorize(lambda x: len(str(x)) == 6 and
                                       bool(regex.match(str(x))))
        if vectorize_regex(s) == 1:
            return 0
        else:
            return 1

    date_test_str = F.udf(lambda s: is_date_object(s), IntegerType())
    return df_spark.select(
        [F.count(
            F.when(date_test_str(date_col) == 1, 1).otherwise(None)
            ).alias(date_col)]).collect()


def check_dtype(name, dtype, variables_nominal, verbose=False):
    """
    Categorized the variables in oridinal, numerical, categorical and
    nominal.
    """
    var_dtype = "other"
    if dtype == "int":
        var_dtype = "ordinal"
    elif dtype == "double" or dtype == "float":
        var_dtype = "numerical"
    elif dtype == "string":
        var_dtype = "categorical"
    if name in variables_nominal:
        if dtype == "string":
            if verbose:
                print("datatype-check: variable {} tagged as nominal"
                      " shall be treated as categorical.".format(name))
            var_dtype = "categorical"
        else:
            var_dtype = "nominal"
    return var_dtype


def groups_cat_rdd(df, variables, special_values):
    """Counts the number of records per category."""
    def map_group_key(row, variables, special):
        vect = []
        for i, elt in enumerate(row):
            if (elt is not None) and (elt not in special):
                vect.append((str(elt) + "$___$" + variables[i], 1))
        return vect
    special_str = [str(elt) for elt in special_values]
    result_groups = df.select(variables).rdd.map(
        lambda x: map_group_key(x, variables, special_str)
        ).flatMap(lambda x: x).countByKey()
    return dict(result_groups)


def histogram_rdd(df_spark, variables, dict_var, special_values):
    """Calculates the number of bins and its boundaries for the histogram."""
    def map_func(row, special, dict_var, variables):
        low_vect = []
        high_vect = []
        for i, elt in enumerate(row):
            if (elt is not None) and (elt not in special):
                if elt < dict_var[variables[i]]["d_outlier_low"]:
                    low_vect.append(1)
                    high_vect.append(0)
                elif elt > dict_var[variables[i]]["d_outlier_high"]:
                    high_vect.append(1)
                    low_vect.append(0)
                else:
                    high_vect.append(0)
                    low_vect.append(0)
            else:
                high_vect.append(0)
                low_vect.append(0)
        return [low_vect, high_vect]

    def reduce_func(vec1, vec2):
        for i in range(len(vec1[0])):
            # count
            vec1[0][i] += vec2[0][i]  # count low outliers
            vec1[1][i] += vec2[1][i]  # count high outliers
        return vec1

    result = df_spark.select(variables).rdd.map(
        lambda x: map_func(x, special_values, dict_var, variables)
        ).reduce(lambda vec1, vec2: reduce_func(vec1, vec2))
    for i, col in enumerate(variables):
        dict_var[col]["d_n_outlier_low"] = result[0][i]
        dict_var[col]["d_n_outlier_high"] = result[1][i]
        if dict_var[col]["iqr"] is not None:
            # binning formula of Sturges. With a cap at 20 bins and floor
            # at 1 bin.
            dict_var[col]["n_bins"] = int(round(min(max(
                1, 1+np.log2(dict_var[col]["n_informed"] -
                             result[0][i] - result[1][i] + 1e-15)), 20)))
            min_val = max(dict_var[col]["d_outlier_low"],
                          dict_var[col]["d_min"])
            max_val = min(dict_var[col]["d_outlier_high"],
                          dict_var[col]["d_max"])
            dict_var[col]["increment"] = (
                max_val - min_val) / dict_var[col]["n_bins"]
            if dict_var[col]["increment"] == 0.0:
                dict_var[col]["n_bins"] = 1
            dict_var[col]["d_hist_pos"] = [
                (min_val + i*dict_var[col]["increment"])
                for i in range(dict_var[col]["n_bins"]+1)]


def histogram2_rdd(df_spark, variables, dict_var, special_values):
    """Calculates the number of records in each histogram's bin."""
    def map_func(row, special, dict_var, variables):
        hist_vect = []
        for i, elt in enumerate(row):
            vect = [0]*dict_var[variables[i]]["n_bins"]
            if (elt is not None) and (elt not in special):
                if (elt >= dict_var[variables[i]]["d_outlier_low"] and
                        elt <= dict_var[variables[i]]["d_outlier_high"]):
                    if dict_var[variables[i]]["increment"] == 0.0:
                        vect[0] += 1
                    else:
                        idx = int((elt -
                                   max(dict_var[variables[i]]["d_outlier_low"],
                                       dict_var[variables[i]]["d_min"])) /
                                  dict_var[variables[i]]["increment"])
                        if idx+1 > len(vect):
                            idx = len(vect) - 1
                        vect[idx] += 1
                    hist_vect.append(vect)
                else:
                    hist_vect.append(vect)
            else:
                hist_vect.append(vect)
        return [hist_vect]

    def reduce_func(vec1, vec2):
        for i in range(len(vec1[0])):
            # count
            for j in range(len(vec1[0][i])):
                vec1[0][i][j] += vec2[0][i][j]
        return vec1

    resultado = df_spark.select(variables).rdd.map(
        lambda x: map_func(x, special_values, dict_var, variables)
        ).reduce(lambda vec1, vec2: reduce_func(vec1, vec2))
    for i, col in enumerate(variables):
        dict_var[col]["d_hist_cols"] = resultado[0][i]
        if dict_var[col]["n_informed"] > 0:
            idx_concentration = np.argmax(dict_var[col]["d_hist_cols"])
            dict_var[col]["d_concentration"] = (
                dict_var[col]["d_hist_cols"][idx_concentration] /
                dict_var[col]["n_informed"])
            dict_var[col]["d_concentration_interval"] = (
                dict_var[col]["d_hist_pos"][idx_concentration],
                dict_var[col]["d_hist_pos"][idx_concentration + 1])
        else:
            dict_var[col]["d_concentration"] = 0
            dict_var[col]["d_concentration_interval"] = (0, 0)


def metrics_cat_rdd(df, cols, special_values):
    """Calculates the main metrics for categorical variables."""
    def map_func(row, special):
        nan_bool = []
        special_bool = []
        infor_bool = []
        for elt in row:
            if elt is None:
                nan_bool.append(1)
                special_bool.append(0)
                infor_bool.append(0)
            elif elt in special:
                nan_bool.append(0)
                special_bool.append(1)
                infor_bool.append(0)
            else:
                nan_bool.append(0)
                special_bool.append(0)
                infor_bool.append(1)
        return [nan_bool, special_bool, infor_bool]

    def reduce_func(vec1, vec2):

        for i in range(len(vec1[0])):
            # count
            vec1[0][i] += vec2[0][i]  # count nan
            vec1[1][i] += vec2[1][i]  # count special
            vec1[2][i] += vec2[2][i]  # count informed
        return vec1
    special_str = [str(elt) for elt in special_values]
    resultado = df.select(cols).rdd.map(
        lambda x: map_func(x, special_str)
        ).reduce(lambda vec1, vec2: reduce_func(vec1, vec2))
    return resultado


def metrics_numerical_rdd(df, cols, special_values):
    """Calculates the main metrics for numerical variables."""
    def map_func(row, special):
        nan_bool = []
        special_bool = []
        zeros_bool = []
        negative_bool = []
        infor_bool = []
        mean_vect = []
        M2_vect = []
        min_vect = []
        max_vect = []
        for elt in row:
            M2_vect.append(0)
            if elt is None:
                nan_bool.append(1)
                special_bool.append(0)
                zeros_bool.append(0)
                negative_bool.append(0)
                infor_bool.append(0)
                mean_vect.append(None)
                min_vect.append(None)
                max_vect.append(None)
            elif elt in special:
                nan_bool.append(0)
                special_bool.append(1)
                zeros_bool.append(0)
                negative_bool.append(0)
                infor_bool.append(0)
                mean_vect.append(None)
                min_vect.append(None)
                max_vect.append(None)
            elif elt == 0:
                nan_bool.append(0)
                special_bool.append(0)
                zeros_bool.append(1)
                negative_bool.append(0)
                infor_bool.append(1)
                mean_vect.append(elt)
                min_vect.append(elt)
                max_vect.append(elt)
            elif elt < 0:
                nan_bool.append(0)
                special_bool.append(0)
                zeros_bool.append(0)
                negative_bool.append(1)
                infor_bool.append(1)
                mean_vect.append(elt)
                min_vect.append(elt)
                max_vect.append(elt)
            else:
                nan_bool.append(0)
                special_bool.append(0)
                zeros_bool.append(0)
                negative_bool.append(0)
                infor_bool.append(1)
                mean_vect.append(elt)
                min_vect.append(elt)
                max_vect.append(elt)
        return [nan_bool, special_bool, zeros_bool,
                negative_bool, infor_bool, mean_vect, M2_vect,
                min_vect, max_vect]

    def reduce_func(vec1, vec2):

        for i in range(len(vec1[0])):
            # count
            vec1[0][i] += vec2[0][i]  # count nan
            vec1[1][i] += vec2[1][i]  # count special
            vec1[2][i] += vec2[2][i]  # count 0
            vec1[3][i] += vec2[3][i]  # count <0
            if vec1[4][i] == 0:
                vec1[4][i] += vec2[4][i]  # count all informed
                vec1[5][i] = vec2[5][i]  # get the mean of vec2
                vec1[6][i] = vec2[6][i]  # get the std of vec2
                vec1[7][i] = vec2[7][i]  # min value
                vec1[8][i] = vec2[8][i]  # max value
            elif vec2[4][i] != 0:
                # be aware that both vec1 and vec2 may be already combinations
                # therefore simple online algorithm (n-1 --> n) is not valid
                # since is (n-m --> n)
                vec1[4][i] += vec2[4][i]  # count all informed
                # parallel online algorithms.
                mean = vec1[5][i] + ((vec2[5][i] - vec1[5][i]) *
                                     vec2[4][i]) / vec1[4][i]  # mean
                convert_1 = (vec1[4][i]-vec2[4][i]-1)/(vec1[4][i]-1)
                convert_2 = (vec2[4][i]-1)/(vec1[4][i]-1)
                vec1[6][i] = (
                    vec1[6][i]*convert_1 + vec2[6][i]*convert_2 +
                    (vec1[5][i]-vec2[5][i])**2 *
                    ((vec1[4][i]-vec2[4][i])*vec2[4][i]) /
                    (vec1[4][i]*(vec1[4][i]-1)))  # Variance

                vec1[5][i] = mean
                if vec1[7][i] > vec2[7][i]:
                    vec1[7][i] = vec2[7][i]  # min value
                if vec1[8][i] < vec2[8][i]:
                    vec1[8][i] = vec2[8][i]  # max value
        return vec1
    result = df.select(cols).rdd.map(
        lambda x: map_func(x, special_values)
        ).reduce(lambda vec1, vec2: reduce_func(vec1, vec2))
    # change variance to std
    std_vec = []
    for i, elt in enumerate(result[6]):
        if result[4][i] < 2:
            std_vec.append(0)
        else:
            std_vec.append((elt)**(1/2))
    return result + [std_vec]


def time_groups_cat_rdd(df, cols, special_values, dates_name, dict_cat):
    """Calculates the groups population of categorical vars for each date."""
    def map_group_key_time(row, variables, special, dict_time, date_name):
        vect = []
        fecha = row[date_name]
        if fecha is None:
            fecha = "nan"
        for i, elt in enumerate(row):
            if ((elt is not None) and (str(elt) not in special) and
                    (str(elt) in dict_time[variables[i]])):
                vect.append((str(elt) + "$___$" + variables[i] + "$___$" +
                             str(fecha), 1))
            if (i+1) == len(variables):
                return vect

    dict_time_groups = {}
    for key in dict_cat:
        dict_time_groups[key] = [
            elt[0] for elt in dict_cat[key]["10_categories"]]

    special_str = [str(elt) for elt in special_values]
    result_time_cat = df.select(cols+[dates_name]).rdd.map(
        lambda x: map_group_key_time(
            x, cols, special_str, dict_time_groups, dates_name)
        ).flatMap(lambda x: x).countByKey()

    return result_time_cat


def time_metrics_cat_rdd(df, cols, special_values, dates_name):
    """Calculates the main metrics time evolution of categorical vars."""
    def map_func(row, special):
        nan_bool = []
        special_bool = []
        infor_bool = []
        for elt in row:
            if elt is None:
                nan_bool.append(1)
                special_bool.append(0)
                infor_bool.append(0)
            elif elt in special:
                nan_bool.append(0)
                special_bool.append(1)
                infor_bool.append(0)
            else:
                nan_bool.append(0)
                special_bool.append(0)
                infor_bool.append(1)
        return [nan_bool, special_bool, infor_bool]

    def reduce_func(vec1, vec2):
        for i in range(len(vec1[0])-1):
            # count
            vec1[0][i] += vec2[0][i]  # count nan
            vec1[1][i] += vec2[1][i]  # count special
            vec1[2][i] += vec2[2][i]  # count all informed
        return vec1
    special_str = [str(elt) for elt in special_values]
    result = df.select(cols).rdd.map(
        lambda x: (x[dates_name], map_func(x, special_str))
        ).reduceByKey(lambda vec1, vec2: reduce_func(vec1, vec2)).collect()

    return result


def time_metrics_rdd(df, cols, special_values, dates_name):
    """Calculates the main metrics time evolution of numerical vars."""
    def map_func(row, special):
        nan_bool = []
        special_bool = []
        infor_bool = []
        mean_vect = []
        for elt in row:
            mean_vect.append(elt)
            if elt is None:
                nan_bool.append(1)
                special_bool.append(0)
                infor_bool.append(0)
            elif elt in special:
                nan_bool.append(0)
                special_bool.append(1)
                infor_bool.append(0)
            else:
                nan_bool.append(0)
                special_bool.append(0)
                infor_bool.append(1)
        return [nan_bool, special_bool, infor_bool, mean_vect]

    def reduce_func(vec1, vec2):
        for i in range(len(vec1[0])-1):
            # count
            vec1[0][i] += vec2[0][i]  # count nan
            vec1[1][i] += vec2[1][i]  # count special
            if vec1[2][i] == 0:
                vec1[2][i] += vec2[2][i]  # count all informed
                vec1[3][i] = vec2[3][i]  # get the mean of vec2
            elif vec2[2][i] != 0:
                vec1[2][i] += vec2[2][i]  # count all informed
        return vec1

    result = df.select(cols).rdd.map(
        lambda x: (x[dates_name], map_func(x, special_values))
        ).reduceByKey(lambda vec1, vec2: reduce_func(vec1, vec2)).collect()

    return result
