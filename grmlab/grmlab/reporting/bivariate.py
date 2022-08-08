"""
Data analysis bivariate class reporting.
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
		"iv": info["iv"],
		"gini": info["gini"],
		"pvalue_chi2": info["pvalue_chi2"],
		"pvalue_chi2_max": info["pvalue_chi2_max"],
		"cramer_v": info["cramer_v"],
		"pd_monotonicity": info["pd_monotonicity"],
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
			"temp_split_event_by_date": info["t_split_event_by_date"],
			"temp_n_split_empty": info["t_n_split_empty"],
			"temp_n_months_no_event": info["t_n_months_no_event"],
			"temp_pd_mean": info["t_pd_mean"],
			"temp_pd_std": info["t_pd_std"],
			"temp_pd_min": info["t_pd_min"],
			"temp_pd_p25": info["t_pd_percentile_25"],
			"temp_pd_median": info["t_pd_median"],
			"temp_pd_p75": info["t_pd_percentile_75"],
			"temp_pd_max": info["t_pd_max"],
			"temp_pd_dispersion": info["t_pd_dispersion"],
			"temp_iv_mean": info["t_iv_mean"],
			"temp_iv_std": info["t_iv_std"],
			"temp_iv_min": info["t_iv_min"],
			"temp_iv_p25": info["t_iv_percentile_25"],
			"temp_iv_median": info["t_iv_median"],
			"temp_iv_percentile_75": info["t_iv_percentile_75"],
			"temp_iv_max": info["t_iv_max"],
			"temp_split_div_uniform": info["t_split_div_uniform"],
			"temp_split_div_exponential": info["t_split_div_exponential"]
		}
	else:
		bivariate_common_temporal = {
			"dates": [],
			"temp_split_count_by_date": [],
			"temp_split_event_by_date": [],
			"temp_n_split_empty": np.nan,
			"temp_n_months_no_event": np.nan,
			"temp_pd_mean": [],
			"temp_pd_std": [],
			"temp_pd_min": [],
			"temp_pd_p25": [],
			"temp_pd_median": [],
			"temp_pd_p75": [],
			"temp_pd_max": [],
			"temp_pd_dispersion": np.nan,
			"temp_iv_mean": np.nan,
			"temp_iv_std": np.nan,
			"temp_iv_min": np.nan,
			"temp_iv_p25": np.nan,
			"temp_iv_median": np.nan,
			"temp_iv_percentile_75": np.nan,
			"temp_iv_max": np.nan,
			"temp_split_div_uniform": np.nan,
			"temp_split_div_exponential": np.nan
		}

	bivariate_common = {**bivariate_common, **bivariate_common_temporal}

	if info["dtype"] in ("numerical", "ordinal"):
		bivariate_dtype = {
			"pvalue_anova": info["pvalue_anova"],
			"pvalue_kolmogorov_smirnov": info["pvalue_ks"],
			"pvalue_kruskal_wallis": info["pvalue_kw"],
			"pvalue_median_t": info["pvalue_median_t"],
			"divergence": info["divergence"],
			"wasserstein_distance": info["wasserstein_distance"],
			"mean_nonevent": info["mean_nonevent"],
			"std_nonevent": info["std_nonevent"],
			"min_nonevent": info["min_nonevent"],
			"p25_nonevent": info["percentile_25_nonevent"],
			"median_nonevent": info["median_nonevent"],
			"p75_nonevent": info["percentile_75_nonevent"],
			"max_nonevent": info["max_nonevent"],
			"mean_event": info["mean_event"],
			"std_event": info["std_event"],
			"min_event": info["min_event"],
			"p25_event": info["percentile_25_event"],
			"median_event": info["median_event"],
			"p75_event": info["percentile_75_event"],
			"max_event": info["max_event"]
		}
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

	min_iv = bivariate.min_iv
	max_iv = bivariate.max_iv
	min_gini = bivariate.min_gini
	max_gini = bivariate.max_gini
	max_pvalue_anova = bivariate.max_pvalue_anova
	max_pvalue_chi2 = bivariate.max_pvalue_chi2
	max_pvalue_ks = bivariate.max_pvalue_ks
	max_pvalue_kw = bivariate.max_pvalue_kw
	max_pvalue_median_t = bivariate.max_pvalue_median_t
	min_cramer_v = bivariate.min_cramer_v

	monotonicity_force = bivariate.monotonicity_force

	special_handler_policy = bivariate.special_handler_policy
	special_woe_policy = bivariate.special_woe_policy
	missing_woe_policy = bivariate.missing_woe_policy

	verbose = bivariate.verbose

	param_data = {
		"index": ["target", "date", "optimalgrouping", "nominal variables",
			"special values", "min iv", "max iv", "min gini", "max gini",
			"max p-value anova", "max p-value chi2", "max p-value ks",
			"max p-value kw", "max p-value median-t", "min cramer v",
			"monotonicity force", "special handler policy", "special woe policy",
			"missing woe policy", "verbose"],
		"data": [target, date, optimalgrouping, variables_nominal,
			special_values, min_iv, max_iv, min_gini, max_gini,
			max_pvalue_anova, max_pvalue_chi2, max_pvalue_ks, max_pvalue_kw,
			max_pvalue_median_t, min_cramer_v, monotonicity_force,
			special_handler_policy, special_woe_policy, missing_woe_policy,
			verbose]
	}

	param_block = content_blocks_element("parameters", param_data)
	# build param step contents
	content_blocks = [param_block]
	step_contents_config_run = step_contents_element(
		content_position="sidebar_left", content_type="parameters",
		content_blocks=content_blocks)

	# bivariate run statistics
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

	categories = ["low IV", "high IV", "low gini", "high gini",
		"high p-value chi2", "high p-value chi2 max", "low cramer V",
		"high p-value anova", "high p-value KS", "high p-value KW",
		"high p-value median-t", "undefined"]

	status = ["IV_low", "IV_high", "Gini_low", "Gini_high", "pvalue_chi2_high",
		"pvalue_max_chi2_high", "cramer_v_low", "pvalue_anova_high",
		"pvalue_ks_high", "pvalue_kw_high", "pvalue_median_t_high", "undefined"]

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
