"""
Data analysis bivariate continuous class reporting.
"""

import json

import numpy as np

from .util import content_blocks_element
from .util import dict_to_json
from .util import step_contents_element


# JSON code
JSON_DATA_ITEMS = "json_items_data['{}'] = ('{}');"

# reporting messages
AVAILABLE = "available"
NOT_AVAILABLE = "not available"


def add_bivariate_variable(variable, info, item_id, dates):
	# variable information - run step ("01")
	step_id = "01"
	content_position = 2
	block_position = 0

	# binning tables: pandas dataframe ==> json
	json_basic_binning_table = json.loads(
		info["binning_table"].to_json(orient="split", index=False))

	json_optimal_binning_table = json.loads(
		info["optimal_binning_table"].to_json(orient="split", index=False))

	bivariate_common = {
		"name": variable,
		"dtype": info["dtype"],
		"r2_score": info["r2_score"],
		"corr_pearson": info["corr_pearson"],
		"corr_spearman": info["corr_spearman"],
		"corr_kendalltau": info["corr_kendalltau"],
		"pvalue_welch_t": info["pvalue_welch_t"],
		"pvalue_mann_whitney": info["pvalue_mann_whitney"],
		"pvalue_median": info["pvalue_median"],
		"mean_absolute_error": info["mean_absolute_error"],
		"mean_squared_error": info["mean_squared_error"],
		"median_absolute_error": info["median_absolute_error"],
		"monotonicity": info["monotonicity"],
		"groups": info["groups"],
		"group_special": info["group_special"],
		"group_missing": info["group_missing"],
		"group_others": info["group_others"],
		"basic_binning_table": json_basic_binning_table,
		"optimal_binning_table": json_optimal_binning_table	
	}

	if len(dates):
		bivariate_common_temporal = {
			"dates": dates,
			"temp_split_count_by_date": info["t_split_count_by_date"],
			"temp_split_mean_by_date": info["t_split_mean_by_date"],
			"temp_split_median_by_date": info["t_split_median_by_date"],
			"temp_n_split_empty": info["t_n_split_empty"],
			"temp_mean_mean": info["t_mean_mean"],
			"temp_mean_std": info["t_mean_std"],
			"temp_mean_min": info["t_mean_min"],
			"temp_mean_percentile_25": info["t_mean_percentile_25"],
			"temp_mean_median": info["t_mean_median"],
			"temp_mean_percentile_75": info["t_mean_percentile_75"],
			"temp_mean_max": info["t_mean_max"],
			"temp_mean_dispersion": info["t_mean_dispersion"],
			"temp_median_mean": info["t_median_mean"],
			"temp_median_std": info["t_median_std"],
			"temp_median_min": info["t_median_min"],
			"temp_median_percentile_25": info["t_median_percentile_25"],
			"temp_median_median": info["t_median_median"],
			"temp_median_percentile_75": info["t_median_percentile_75"],
			"temp_median_max": info["t_median_max"],
			"temp_median_dispersion": info["t_median_dispersion"],
			"temp_r2_mean": info["t_r2_mean"],
			"temp_r2_std": info["t_r2_std"],
			"temp_r2_min": info["t_r2_min"],
			"temp_r2_percentile_25": info["t_r2_percentile_25"],
			"temp_r2_median": info["t_r2_median"],
			"temp_r2_percentile_75": info["t_r2_percentile_75"],
			"temp_r2_max": info["t_r2_max"],
			"temp_split_div_uniform": info["t_split_div_uniform"],
			"temp_split_div_exponential": info["t_split_div_exponential"]
		}
	else:
		bivariate_common_temporal = {
			"dates": [],
			"temp_split_count_by_date": [],
			"temp_split_mean_by_date": [],
			"temp_split_median_by_date": [],
			"temp_n_split_empty": np.nan,
			"temp_mean_mean": [],
			"temp_mean_std": [],
			"temp_mean_min": [],
			"temp_mean_percentile_25": [],
			"temp_mean_median": [],
			"temp_mean_percentile_75": [],
			"temp_mean_max": [],
			"temp_mean_dispersion": [],
			"temp_median_mean": [],
			"temp_median_std": [],
			"temp_median_min": [],
			"temp_median_percentile_25": [],
			"temp_median_median": [],
			"temp_median_percentile_75": [],
			"temp_median_max": [],
			"temp_median_dispersion": np.nan,
			"temp_r2_mean": np.nan,
			"temp_r2_std": np.nan,
			"temp_r2_min": np.nan,
			"temp_r2_percentile_25": np.nan,
			"temp_r2_median": np.nan,
			"temp_r2_percentile_75": np.nan,
			"temp_r2_max": np.nan,
			"temp_split_div_uniform": np.nan,
			"temp_split_div_exponential": np.nan
		}

	bivariate_common = {**bivariate_common, **bivariate_common_temporal}

	if info["dtype"] in ("numerical", "ordinal"):
		bivariate_dtype = {}
	else:
		bivariate_dtype = {
			"categories": info["n_categories"],
			"most frequent category": info["most_freq_category"],
			"most frequent category (%)": info["p_most_freq_category"],
			"mean_number_categories": info["mean_n_categories"],
			"std_number_categories": info["std_n_categories"],
			"min_number_categories": info["min_n_categories"],
			"p25_number_categories": info["percentile_25_n_categories"],
			"median_number_categories": info["median_n_categories"],
			"p75_number_categories": info["percentile_75_n_categories"],
			"max_number_categories": info["max_n_categories"]
		}

	bivariate_variable = {**bivariate_common, **bivariate_dtype}

	# variable structure
	json_bivariate = {
		"item_id": item_id,
		"step_id": step_id,
		"content_position": content_position,
		"block_position": block_position,
		"data": bivariate_variable
	}

	variable_id = "{}_{}_{}_{}_{}".format(item_id, step_id,
		content_position, block_position, variable)

	str_json_variable = JSON_DATA_ITEMS.format(variable_id,
		dict_to_json(json_bivariate))

	return str_json_variable


def add_bivariate(bivariate, item_title="Bivariate Analysis"):
	# bivariate parameters
	target = bivariate.target

	if bivariate.date is None:
		date = NOT_AVAILABLE
	else:
		date = bivariate.date

	if bivariate.optimalgrouping is None:
		optimalgrouping = NOT_AVAILABLE
	else:
		optimalgrouping = AVAILABLE

	if not bivariate.special_values:
		special_values = NOT_AVAILABLE
	else:
		special_values = AVAILABLE

	if not bivariate.variables_nominal:
		variables_nominal = NOT_AVAILABLE
	else:
		variables_nominal = AVAILABLE

	min_r2_score = bivariate.min_r2_score
	min_corr_pearson = bivariate.min_corr_pearson
	min_corr_spearman = bivariate.min_corr_spearman
	min_corr_kendalltau = bivariate.min_corr_kendalltau
	max_pvalue_welch_t = bivariate.max_pvalue_welch_t
	max_pvalue_mann_whitney = bivariate.max_pvalue_mann_whitney
	max_pvalue_median = bivariate.max_pvalue_median

	monotonicity_force = bivariate.monotonicity_force
	metric = bivariate.metric

	special_handler_policy = bivariate.special_handler_policy

	verbose = bivariate.verbose

	param_data = {
		"index": ["target", "date", "optimalgrouping", "nominal variables",
			"special values", "max p-value Welcht-t",
			"max p-value Mann Whitney", "max p-value median-t",
			"min corr Pearson", "min corr Spearman", "min corr Kendall tau",
			"min R^2 score", "motonicity force", "metric",
			"special handler policy", " verbose"],
		"data": [target, date, optimalgrouping, variables_nominal,
		special_values, max_pvalue_welch_t, max_pvalue_mann_whitney,
		max_pvalue_median, min_corr_pearson, min_corr_spearman,
		min_corr_kendalltau, min_r2_score, monotonicity_force, metric,
		special_handler_policy, verbose]
	}

	param_block = content_blocks_element("parameters", param_data)
	# build param step contents
	content_blocks = [param_block]
	step_contents_config_run = step_contents_element(
		content_position="sidebar_left", content_type="parameters",
		content_blocks=content_blocks)

	# bivarate run statistics
	db_info_data = {
		"index": ["samples", "variables", "date variable", "target variable",
			"target dtype"],
		"data": [bivariate._n_samples, bivariate._n_vars,
			bivariate.date, bivariate.target, bivariate._target_dtype]
	}

	db_info_block = content_blocks_element("db_info_expanded", db_info_data)

	# variable analysis
	dict_values = bivariate._dict_variables.values()

	dtypes = ["numerical", "ordinal", "categorical", "nominal"]

	categories = ["low R^2", "low corr. Pearson", "low corr. Spearman",
		"low corr. Kendall tau", "high p-value Welch-t",
		"high p-value Mann-Whitney", "high p-value median-t", "undefined"]

	status = ["r2_score_low", "corr_pearson_low", "corr_spearman_low",
		"corr_kendalltau_low", "pvalue_welch_t_high", "pvalue_mann_whitney_high",
		"pvalue_median_high", "undefined"]

	dtypes_data = [sum(d["dtype"] == c for d in dict_values)
		for c in dtypes]

	categories_data = [sum(c in d["status"] for d in dict_values)
		for c in status]

	column_analysis_data = {
		"index": dtypes + categories,
		"data": dtypes_data + categories_data
	}

	column_analysis_block = content_blocks_element("column_analysis",
	column_analysis_data)	

	# cpu time
	cpu_time_data = {
		"index": ["total", "optimalgrouping", "metrics"],
		"data": [bivariate._time_run, bivariate._time_run_optimalgrouping,
			bivariate._time_run_metrics]
	}

	cpu_time_metrics = {
		"index": ["total", "numerical", "ordinal", "categorical", "nominal"],
		"data": [bivariate._time_run_metrics, bivariate._time_run_numerical,
			bivariate._time_run_ordinal, bivariate._time_run_categorical,
			bivariate._time_run_nominal]
	}

	cpu_time_block = content_blocks_element("cpu_time", cpu_time_data)
	cpu_time_metrics_block = content_blocks_element("cpu_time", cpu_time_metrics)

	# build stats step contents
	content_blocks = [db_info_block	, column_analysis_block, cpu_time_block,
		cpu_time_metrics_block]

	step_contents_stats_run = step_contents_element(
		content_position="sidebar_left", content_type="stats",
		content_blocks=content_blocks)

	# bivariate transform statistics
	if bivariate._is_transformed:
		param_data = {
			"index": ["mode"],
			"data": [bivariate._transform_mode]
		}

		param_block = content_blocks_element("parameters", param_data)
		# build param step contents
		content_blocks = [param_block]
		step_contents_config_transform = step_contents_element(
			content_position="sidebar_left", content_type="parameters",
			content_blocks=content_blocks)

		# bivariate transform statistics
		n_variables_after = bivariate._n_vars - bivariate._n_vars_remove

		n_user_actions = sum(1 for d in dict_values if "user_action" in d)
		n_user_comment = sum(d["user_comment"] != "" for d in dict_values
			if "user_action" in d)

		results_data = {
			"index": ["original data", "after bivariate", "removed",
				"user actions", "user_comments"],
			"data": [bivariate._n_vars, n_variables_after,
				bivariate._n_vars_remove, n_user_actions, n_user_comment]
		}

		results_block = content_blocks_element("results", results_data)

		# cpu time
		cpu_time_data = {
			"index": ["total", "remove"],
			"data": [bivariate._time_transform, bivariate._time_transform]
		}

		content_blocks = [results_block, cpu_time_block]
		step_contents_stats_transform = step_contents_element(
			content_position="sidebar_left", content_type="stats",
			content_blocks=content_blocks)
	else:
		step_contents_config_transform = None
		step_contents_stats_transform = None	

	item_info = {
		"grmlabcls": bivariate,
		"item_title": item_title,
		"step_config_run": step_contents_config_run,
		"step_config_transform": step_contents_config_transform,
		"step_stats_run": step_contents_stats_run,
		"step_stats_transform": step_contents_stats_transform,
		"results_run_names": None,
		"results_transform_names": None,
		"extra_item_step": None
	}

	return item_info
