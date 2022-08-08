"""
Model Analysis class reporting.
"""

# Authors: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2019.

import numpy as np

from .util import content_blocks_element
from .util import dict_to_json
from .util import step_contents_element


# JSON code
JSON_DATA_ITEMS = "json_items_data['{}'] = ('{}');"


def add_corr_analysis(info, item_id):
    # variable information - run step ("01")
    step_id = "01"
    content_position = 2
    block_position = 0

    # dict mds
    mds_dict = {}
    for var in info._column_names:
        mds_dict[var] = {"vec_0": None, "vec_1": None, "weight": None}
        if info._dict_variables[var]["mds"] is not None:
            mds_dict[var]["vec_0"] = info._dict_variables[var]["mds"][0]
            mds_dict[var]["vec_1"] = info._dict_variables[var]["mds"][1]
        mds_dict[var]["weight"] = info._dict_variables[var]["weight"]
        mds_dict[var]["group_id"] = info._dict_variables[var]["group_id"]
        mds_dict[var]["excluded"] = info._dict_variables[var]["excluded"]
        mds_dict[var]["excluded_by"] = info._dict_variables[var]["excluded_by"]
        mds_dict[var]["connections"] = info._dict_variables[var]["connections"]

    # weight
    if info._flag_weights:
        weight_dict = {"index": list(info.metric_weights.keys()),
                       "data": list(info.metric_weights.values())}
    else:
        weight_dict = {"index": [], "data": []}

    # all info of model analysis inputs and characteristics
    analyzer_metrics = {
        "name": info.name,
        "parameters": {
            "index": ["correlation_method", "equal_mincorr", "groups_mincorr",
                      "links_mincorr"],
            "data": [info.correlation_method, info.equal_mincorr,
                     info.groups_mincorr, info.links_mincorr],
            "weight": weight_dict
        },
        "correlations": {
            "info": list(info._column_distinct),
            "data": [info._dict_variables[var]["corr"]
                     for var in info._column_distinct]
        },
        "mds": mds_dict,
        "groups": info._groups,
        "corr_statistics": info._correlation_metrics,
        "df_input": {
            "index": ["samples", "columns"],
            "data": [info._n_samples, info._n_columns]},
    }

    # variable structure
    json_analysis = {
        "item_id": item_id,
        "step_id": step_id,
        "content_position": content_position,
        "block_position": block_position,
        "data": analyzer_metrics
    }

    analysis_data = content_blocks_element("analysis_data", json_analysis)
    step_contents = step_contents_element(
        content_position="", content_type="",
        content_blocks=[analysis_data])

    return step_contents


def add_corr(info):

    # dict mds
    mds_dict = {}
    for var in info._column_names:
        mds_dict[var] = {"vec_0": None, "vec_1": None, "weight": None}
        if info._dict_variables[var]["mds"] is not None:
            mds_dict[var]["vec_0"] = info._dict_variables[var]["mds"][0]
            mds_dict[var]["vec_1"] = info._dict_variables[var]["mds"][1]
        mds_dict[var]["weight"] = info._dict_variables[var]["weight"]
        mds_dict[var]["group_id"] = info._dict_variables[var]["group_id"]
        mds_dict[var]["excluded"] = info._dict_variables[var]["excluded"]
        mds_dict[var]["excluded_by"] = info._dict_variables[var]["excluded_by"]
        mds_dict[var]["connections"] = info._dict_variables[var]["connections"]

    # weight
    if info._flag_weights:
        weight_dict = {"index": list(info.metric_weights.keys()),
                       "data": list(info.metric_weights.values())}
    else:
        weight_dict = {"index": [], "data": []}

    # all info of model analysis inputs and characteristics
    analyzer_metrics = {
        "name": info.name,
        "parameters": {
            "index": ["correlation_method", "equal_mincorr", "groups_mincorr",
                      "links_mincorr"],
            "data": [info.correlation_method, info.equal_mincorr,
                     info.groups_mincorr, info.links_mincorr],
            "weight": weight_dict
        },
        "correlations": {
            "info": list(info._column_distinct),
            "data": [info._dict_variables[var]["corr"]
                     for var in info._column_distinct]
        },
        "mds": mds_dict,
        "groups": info._groups,
        "corr_statistics": info._correlation_metrics,
        "df_input": {
            "index": ["samples", "columns"],
            "data": [info._n_samples, info._n_columns]},
    }

    return analyzer_metrics


def add_multivariate_variable(variable, info, item_id, multivariate):
    # variable information - run step ("01")
    step_id = "01"
    content_position = 2
    block_position = 0

    # features
    regression_analysis = dict()
    regression_analysis = {
        "info": ["name", "coeff.", "vif", "corr", "var_explained"],
        "data": []}
    for corr_var in info['coef']:
        if (info['coef'][corr_var] != 0):
            if corr_var in multivariate.unique_groups:
                vif = multivariate._dict_groups[corr_var]['vif']
            else:
                vif = multivariate._dict_variables[corr_var]['vif']
            coef = info['coef'][corr_var]
            corr = multivariate._pca_corr[variable][corr_var]
            var_explained = info['var_explained'][corr_var]
            regression_analysis["data"].append([
                corr_var, coef, vif, corr, var_explained])
    regression_analysis["data"] = sorted(
        regression_analysis["data"], key=lambda x: x[4], reverse=True)

    correlation_analysis = dict()
    correlation_analysis = {
        "info": ["name", "corr", "group_id", "weight", "excluded"],
        "data": []}
    for corr_var in multivariate.mcorr.df_corr.columns:
        if corr_var == variable:
            continue
        corr = multivariate.mcorr.df_corr[variable][corr_var]
        if (corr >= multivariate.links_mincorr):
            group_id = multivariate._dict_variables[corr_var]["group_id"]
            weight = multivariate._dict_variables[corr_var]["weight"]
            excluded = multivariate._dict_variables[corr_var]["excluded"]
            correlation_analysis["data"].append([
                corr_var, corr, group_id, weight, excluded])
    correlation_analysis["data"] = sorted(
        correlation_analysis["data"], key=lambda x: x[1], reverse=True)

    bivariate_common = {
        "name": variable,
        "dtype": "variable",
        "group_id": info["group_id"],
        "vif": info["vif"],
        "weight": info["weight"],
        "connections": info["connections"],
        "excluded": info["excluded"],
        "excluded_by": info["excluded_by"],
        "regression":  regression_analysis,
        "correlation": correlation_analysis
    }

    bivariate_variable = {**bivariate_common}

    # variable structure
    json_bivariate = {
        "item_id": item_id,
        "step_id": step_id,
        "content_position": content_position,
        "block_position": block_position,
        "data": bivariate_variable
    }

    variable_id = "{}_{}_{}_{}_{}".format(
        item_id, step_id, content_position, block_position, variable)

    str_json_variable = JSON_DATA_ITEMS.format(
        variable_id, dict_to_json(json_bivariate))

    return str_json_variable


def add_multivariate_group(group, info, item_id, multivariate):
    # group information - run step ("01")
    step_id = "01"
    content_position = 2
    block_position = 1

    # Group Resgression results
    regression_analysis = dict()
    regression_analysis = {
        "info": ["name", "coeff.", "vif", "corr", "var_explained"],
        "data": []}
    for corr_var in info['coef']:
        if (info['coef'][corr_var] != 0):
            if corr_var in multivariate.unique_groups:
                vif = multivariate._dict_groups[corr_var]['vif']
            else:
                vif = multivariate._dict_variables[corr_var]['vif']
            coef = info['coef'][corr_var]
            corr = multivariate._pca_corr[group][corr_var]
            var_explained = info['var_explained'][corr_var]
            regression_analysis["data"].append([
                corr_var, coef, vif, corr, var_explained])
    regression_analysis["data"] = sorted(
        regression_analysis["data"], key=lambda x: x[4], reverse=True)

    # Group Components
    components_group = dict()
    components_group = {
        "info": ["name", "weight", "selected", "pca", "lat_1", "lat_2"],
        "data": []}
    pca_weights_vec = np.power(info["pca"].components_[0], 2)
    fa_1_vec = info["fa"].components_[0]
    fa_2_vec = info["fa"].components_[1]
    for i, var in enumerate(info['variables']):
        weight = multivariate._dict_variables[var]['weight']
        if var in info["selected_variables"]:
            selected = True
        else:
            selected = False
        components_group["data"].append(
            [var, weight, selected, pca_weights_vec[i],
             fa_1_vec[i], fa_2_vec[i]])
    components_group["data"] = sorted(
        components_group["data"], key=lambda x: x[1], reverse=True)

    # MDS & correlations
    correlations = {"info": list(info['variables']),
                    "data": []}
    df_group_corr = multivariate.mcorr.df_corr[
        info['variables']].T[info['variables']]
    mds_dict = {}
    for var in info['variables']:
        # correlations
        correlations["data"].append(list(df_group_corr[var].values))
        # MDS
        mds_dict[var] = {"vec_0": None, "vec_1": None, "weight": None}
        mds_dict[var]["vec_0"] = info["mds"][var][0]
        mds_dict[var]["vec_1"] = info["mds"][var][1]
        mds_dict[var]["weight"] = multivariate._dict_variables[var]['weight']

    bivariate_common = {
        "name": group,
        "dtype": "group",
        "groups_mincorr": multivariate.groups_mincorr,
        "links_mincorr": multivariate.links_mincorr,
        "vif": info["vif"],
        "max_weight": info["max_weight"],
        "pca_explainded": info["pca_explainded"],
        "selected_variables": info["selected_variables"],
        "n_vars": info["n_vars"],
        "components":  components_group,
        "regression":  regression_analysis,
        "correlations": correlations,
        "mds": mds_dict
    }

    bivariate_variable = {**bivariate_common}

    # group structure
    json_bivariate = {
        "item_id": item_id,
        "step_id": step_id,
        "content_position": content_position,
        "block_position": block_position,
        "data": bivariate_variable
    }

    variable_id = "{}_{}_{}_{}_{}".format(
        item_id, step_id, content_position, block_position, group)

    str_json_variable = JSON_DATA_ITEMS.format(
        variable_id, dict_to_json(json_bivariate))

    return str_json_variable


def add_multivariate(multivariate, item_title="Multivariate Analysis"):
    # multivariate parameters
    target = multivariate.target

    correlation_method = multivariate.correlation_method

    equal_mincorr = multivariate.equal_mincorr
    groups_mincorr = multivariate.groups_mincorr
    links_mincorr = multivariate.links_mincorr

    if multivariate.metric_weights is not None:
        metrics_vars = list(multivariate.metric_weights.keys())
        metrics_weight = list(multivariate.metric_weights.values())
    else:
        metrics_vars = None
        metrics_weight = None

    max_vif = multivariate.max_vif
    alpha = multivariate.alpha
    n_max_features = multivariate.n_max_features
    max_correlation_target = multivariate.max_correlation_target
    excluded = len(multivariate.excluded)
    fixed = len(multivariate.fixed)

    verbose = multivariate.verbose

    param_data = {
        "index": ["max_vif", "metric_weights(keys)", "metric_weights(values)",
                  "correlation_method", "equal_mincorr",
                  "groups_mincorr", "links_mincorr",
                  "alpha", "n_max_features", "max_correlation_target",
                  "excluded", "fixed", "verbose"],
        "data": [max_vif, metrics_vars, metrics_weight, correlation_method,
                 equal_mincorr, groups_mincorr,
                 links_mincorr, alpha, n_max_features,
                 max_correlation_target, excluded, fixed, verbose],
        "data_corr": add_corr(multivariate.mcorr)
    }

    param_block = content_blocks_element("parameters", param_data)
    # build param step contents
    content_blocks = [param_block]
    step_contents_config_run = step_contents_element(
        content_position="sidebar_left", content_type="parameters",
        content_blocks=content_blocks)

    # bivariate run statistics
    db_info_data = {
        "index": ["samples", "variables", "date variable", "target variable"],
        "data": [multivariate.mcorr._n_samples,
                 multivariate.mcorr._n_columns, multivariate.date, target]
    }

    db_info_block = content_blocks_element("db_info_expanded", db_info_data)

    # variable analysis
    dict_values = multivariate._dict_variables.values()

    dtypes = ["vars_with_group", "vars_without_group", "n_groups"]

    categories = ["high VIF", "high gr. VIF", "excluded", "not selected"]

    status = ["VIF_high", "Group_VIF_high", "Excluded", "Not_selected"]

    dtypes_data = [
        sum(d["group_id"] != "no_group" for d in dict_values),
        sum(d["group_id"] == "no_group" for d in dict_values),
        len(multivariate._dict_groups.keys())]

    categories_data = [sum(c in d["status"] for d in dict_values)
                       for c in status]

    column_analysis_data = {
        "index": dtypes + categories,
        "data": dtypes_data + categories_data
    }

    column_analysis_block = content_blocks_element(
        "column_analysis", column_analysis_data)

    # time
    cpu_time_run = {
        "index": ["total", "corr", "pca/fa", "vif", "group"],
        "data": [multivariate._time_run, multivariate._time_run_corr,
                 multivariate._time_run_pca,
                 multivariate._time_run_vif,
                 multivariate._time_run_group]
    }

    cpu_time_run_block = content_blocks_element("cpu_time", cpu_time_run)

    # build stats step contents
    content_blocks = [db_info_block, column_analysis_block, cpu_time_run_block]

    step_contents_stats_run = step_contents_element(
        content_position="sidebar_left", content_type="stats",
        content_blocks=content_blocks)

    # univariate transform statistics
    if multivariate._is_transformed:
        param_data = {
            "index": ["mode"],
            "data": [multivariate._transform_mode]
        }

        param_block = content_blocks_element("parameters", param_data)
        # build param step contents
        content_blocks = [param_block]
        step_contents_config_transform = step_contents_element(
            content_position="sidebar_left", content_type="parameters",
            content_blocks=content_blocks)

        # univariate transform statistics
        n_variables_after = (multivariate.mcorr._n_columns -
                             multivariate._n_vars_remove)

        dict_values = multivariate._dict_variables.values()

        n_user_actions = sum(d["user_action"] is not None for d in dict_values)
        n_user_comment = sum(d["user_comment"] != ""
                             for d in dict_values
                             if d["user_comment"] is not None)

        results_data = {
            "index": ["original data", "after multivariate", "removed",
                      "user actions", "user_comments"],
            "data": [multivariate.mcorr._n_columns, n_variables_after,
                     multivariate._n_vars_remove, n_user_actions,
                     n_user_comment]
        }

        results_block = content_blocks_element("results", results_data)

        # cpu time
        cpu_time_data = {
            "index": ["total", "remove"],
            "data": [multivariate._time_transform,
                     multivariate._time_transform]
        }

        cpu_time_block = content_blocks_element("cpu_time", cpu_time_data)

        content_blocks = [results_block, cpu_time_block]
        step_contents_stats_transform = step_contents_element(
            content_position="sidebar_left", content_type="stats",
            content_blocks=content_blocks)
    else:
        step_contents_config_transform = None
        step_contents_stats_transform = None

    item_info = {
        "grmlabcls": multivariate,
        "item_title": item_title,
        "step_config_run": step_contents_config_run,
        "step_config_transform": step_contents_config_transform,
        "step_stats_run": step_contents_stats_run,
        "step_stats_transform": step_contents_stats_transform,
        "results_run_names": ["variables", "groups"],
        "results_transform_names": None,
        "extra_item_step": None
    }

    return item_info
