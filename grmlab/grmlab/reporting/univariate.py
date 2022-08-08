"""
Data analysis univariate class reporting.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import numpy as np

from .util import content_blocks_element
from .util import dict_to_json
from .util import step_contents_element


# JSON code
JSON_DATA_ITEMS = "json_items_data['{}'] = ('{}');"

# reporting messages
AVAILABLE = "available"
NOT_AVAILABLE = "not available"


def add_univariate_variable(variable, info, item_id, dates):
	# variable information - run step ("01")
	step_id = "01"
	content_position = 2

	univariate_common = {
		"name": variable,
		"dtype": info["dtype"],
		"informed records": info["n_informed"],
		"missing records": info["n_missing"],
		"special records": info["n_special"],
		"unique special values": info["u_special"],
		"hist_info_col": info["d_hist_cols"],
		"hist_info_pos": info["d_hist_pos"],
	}

	if len(dates):
		univariate_common_temporal = {
			"dates": dates,
			"months w/o information": info["t_n_months_no_info"],
			"information divergence (uniform)": info["t_info_div_uniform"],
			"information divergence (exponential)": info["t_info_div_exponential"],
			"data values divergence (uniform)": info["t_data_div_uniform"],
			"data values divergence (exponential)": info["t_data_div_exponential"],
			"temp_missing": info["t_missing"],
			"temp_special": info["t_special"],
			"temp_info": info["t_info"]
		}
	else:
		univariate_common_temporal = {
			"dates": [],
			"months w/o information": np.nan,
			"information divergence (uniform)": np.nan,
			"information divergence (exponential)": np.nan,
			"data values divergence (uniform)": np.nan,
			"data values divergence (exponential)": np.nan,
			"temp_missing": [],
			"temp_special": [],
			"temp_info": []
		}

	univariate_common = {**univariate_common, **univariate_common_temporal}

	if info["dtype"] in ("numerical", "ordinal"):
		block_position = 0
		univariate_dtype = {
			"zeros": info["d_n_zero"],
			"zeros (%)": info["d_p_zero"],
			"negative": info["d_n_neg"],
			"positive": info["d_n_pos"],
			"min": info["d_min"],
			"max": info["d_max"],
			"mean": info["d_mean"],
			"std": info["d_std"],
			"mode": info["d_mode"],
			"percentile 1%": info["d_percentile_1"],
			"percentile 25%": info["d_percentile_25"],
			"median": info["d_median"],
			"percentile 75%": info["d_percentile_75"],
			"percentile 99%": info["d_percentile_99"],
			"outliers low": info["d_n_outlier_low"],
			"outliers high": info["d_n_outlier_high"],
			"outlier low threshold": info["d_outlier_low"],
			"outlier high threshold": info["d_outlier_high"],
			"concentration": info["d_concentration"],
			"concentration_interval": info["d_concentration_interval"],
			"dispersion_parametric": info["d_coeff_disp_param"],
			"dispersion_nonparametric": info["d_coeff_disp_nonparam"]
		}

		if len(dates):
			univariate_temporal = {
				"temp_mean": info["t_mean"],
				"temp_percentile25": info["t_p25"],
				"temp_median": info["t_median"],
				"temp_percentile75": info["t_p75"]
			}
		else:
			univariate_temporal = {
				"temp_mean": np.nan,
				"temp_percentile25": np.nan,
				"temp_median": np.nan,
				"temp_percentile75": np.nan
			}

		univariate_dtype = {**univariate_dtype, **univariate_temporal}
	else:
		block_position = 1
		univariate_dtype = {
			"categories": info["d_n_categories"],
			"most frequent category": info["d_most_freq_category"],
			"most frequent category (%)": info["d_p_most_freq_category"],
			"HHI": info["d_hhi"],
			"HHI (normalized)": info["d_hhi_norm"],
			"top_cat": info["d_top_categories"],
			"top_cat_n": info["d_n_top_categories"]
		}

		if len(dates):
			univariate_temporal = {
				"temp_c0": info["t_c0"],
				"temp_c1": info["t_c1"],
				"temp_c2": info["t_c2"],
				"temp_c3": info["t_c3"],
				"temp_c4": info["t_c4"],
				"temp_c5": info["t_c5"],
				"temp_c6": info["t_c6"],
				"temp_c7": info["t_c7"],
				"temp_c8": info["t_c8"],
				"temp_c9": info["t_c9"],
				"temp_rest": info["t_rest"]
			}
		else:
			univariate_temporal = {
				"temp_c0": [],
				"temp_c1": [],
				"temp_c2": [],
				"temp_c3": [],
				"temp_c4": [],
				"temp_c5": [],
				"temp_c6": [],
				"temp_c7": [],
				"temp_c8": [],
				"temp_c9": [],
				"temp_rest": []
			}

		univariate_dtype = {**univariate_dtype, **univariate_temporal}

	univariate_variable = {**univariate_common, **univariate_dtype}

	# variable structure
	json_univariate = {
		"item_id": item_id,
		"step_id": step_id,
		"content_position": content_position,
		"block_position": block_position,
		"data": univariate_variable
	}

	variable_id = "{}_{}_{}_{}_{}".format(item_id, step_id,
		content_position, block_position, variable)

	str_json_variable = JSON_DATA_ITEMS.format(variable_id,
		dict_to_json(json_univariate))

	return str_json_variable


def add_univariate(univariate, item_title="Univariate Analysis"):
	# univariate parameters
	if univariate.target is None:
		target = NOT_AVAILABLE
	else:
		target = univariate.target

	if univariate.date is None:
		date = NOT_AVAILABLE
	else:
		date = univariate.date

	if not univariate.special_values:
		special_values = NOT_AVAILABLE
	else:
		special_values = AVAILABLE

	if not univariate.variables_nominal:
		variables_nominal = NOT_AVAILABLE
	else:
		variables_nominal = AVAILABLE

	max_p_missing = univariate.max_p_missing
	max_p_special = univariate.max_p_special
	max_divergence = univariate.max_divergence

	verbose = univariate.verbose

	param_data = {
		"index": ["target", "date", "nominal variables", "special values",
			"max % missing", "max % special", "max divergence", "verbose"],
		"data": [target, date, variables_nominal, special_values, max_p_missing,
			max_p_special, max_divergence, verbose]
	}

	param_block = content_blocks_element("parameters", param_data)
	# build param step contents
	content_blocks = [param_block]
	step_contents_config_run = step_contents_element(
		content_position="sidebar_left", content_type="parameters",
		content_blocks=content_blocks)

	# univariate run statistics
	db_info_data = {
		"index": ["samples", "variables", "date variable", "target variable",
			"target dtype"],
		"data": [univariate._n_samples, univariate._n_vars,
			univariate.date, univariate.target, univariate._target_dtype]
	}

	db_info_block = content_blocks_element("db_info_expanded", db_info_data)

	# variable analysis
	dict_values = univariate._dict_variables.values()

	dtypes = ["numerical", "ordinal", "categorical", "nominal"]
	categories = ["high missing", "high special", "high divergence info",
		"high divergence data"]

	status = ["missing", "special", "info_divergence", "data_divergence"]

	dtypes_data = [sum(d["dtype"] == c for d in dict_values)
		for c in dtypes]

	categories_data = [sum(d["status"] == c for d in dict_values)
		for c in status]

	column_analysis_data = {
		"index": dtypes + categories,
		"data": dtypes_data + categories_data
	}

	column_analysis_block = content_blocks_element("column_analysis",
		column_analysis_data)

	# cpu time
	cpu_time_data = {
		"index": ["total", "numerical", "ordinal", "categorical", "nominal"],
		"data": [univariate._time_run, univariate._time_run_numerical,
			univariate._time_run_ordinal, univariate._time_run_categorical,
			univariate._time_run_nominal]
	}

	cpu_time_block = content_blocks_element("cpu_time", cpu_time_data)

	# build stats step contents
	content_blocks = [db_info_block	, column_analysis_block, cpu_time_block]

	step_contents_stats_run = step_contents_element(
		content_position="sidebar_left", content_type="stats",
		content_blocks=content_blocks)

	# univariate transform statistics
	if univariate._is_transformed:
		param_data = {
			"index": ["mode"],
			"data": [univariate._transform_mode]
		}

		param_block = content_blocks_element("parameters", param_data)
		# build param step contents
		content_blocks = [param_block]
		step_contents_config_transform = step_contents_element(
			content_position="sidebar_left", content_type="parameters",
			content_blocks=content_blocks)

		# univariate transform statistics
		n_variables_after = univariate._n_vars - univariate._n_vars_remove

		n_user_actions = sum(1 for d in dict_values if "user_action" in d)
		n_user_comment = sum(d["user_comment"] != "" for d in dict_values
			if "user_action" in d)

		results_data = {
			"index": ["original data", "after univariate", "removed",
				"user actions", "user_comments"
				],
			"data": [univariate._n_vars, n_variables_after,
				univariate._n_vars_remove, n_user_actions, n_user_comment]
		}

		results_block = content_blocks_element("results", results_data)

		# cpu time
		cpu_time_data = {
			"index": ["total", "remove"],
			"data": [univariate._time_transform, univariate._time_transform]
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
		"grmlabcls": univariate,
		"item_title": item_title,
		"step_config_run": step_contents_config_run,
		"step_config_transform": step_contents_config_transform,
		"step_stats_run": step_contents_stats_run,
		"step_stats_transform": step_contents_stats_transform,
		"results_run_names": ["table_num_ord", "table_cat_nom"],
		"results_transform_names": None,
		"extra_item_step": None
	}

	return item_info