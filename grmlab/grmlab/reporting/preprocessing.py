"""
Data preprocessing class reporting.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

from .util import content_blocks_element
from .util import step_contents_element


# reporting messages
AVAILABLE = "available"
NOT_AVAILABLE = "not available"


def add_preprocessing(preprocessing, item_title="Data Preprocessing"):
	# preprocessing parameters
	if preprocessing.target is None:
		target = NOT_AVAILABLE
	else:
		target = preprocessing.target

	if preprocessing.date is None:
		date = NOT_AVAILABLE
	else:
		date = preprocessing.date

	if not preprocessing.special_values:
		special_values = NOT_AVAILABLE
	else:
		special_values = AVAILABLE

	verbose = preprocessing.verbose

	param_data = {
		"index": ["target", "date", "special values", "verbose"],
		"data": [target, date, special_values, verbose]
	}

	param_block = content_blocks_element("parameters", param_data)
	# build param step contents
	content_blocks = [param_block]
	step_contents_config_run = step_contents_element(
		content_position="sidebar_left", content_type="parameters",
		content_blocks=content_blocks)

	# preprocessing run statistics
	db_info_data = {
		"index": ["samples", "columns"],
		"data": [preprocessing._n_samples, preprocessing._n_columns]
	}

	db_info_block = content_blocks_element("db_info", db_info_data)

	# column analysis
	dict_values = preprocessing._dict_column.values()

	categories = ["binary", "constant", "date", "empty", "exclude", "id",
			"nan_unique", "numeric_conversion", "ok", "special",
			"special_constant", "special_unique"]

	categories_data = [sum([d["status"] == c for d in dict_values])
		for c in categories]

	column_analysis_data = {
		"index": categories,
		"data": categories_data
	}

	column_analysis_block = content_blocks_element("column_analysis",
		column_analysis_data)

	# duplicates analysis
	n_duplicates = sum([d["duplicate"] for d in dict_values])
	n_duplicates_of = sum([1 if d["duplicate_of"] is not None else 0
		for d in dict_values])

	duplicates_analysis_data = {
		"index": ["duplicates", "duplicates_of"],
		"data": [n_duplicates, n_duplicates_of]
	}

	duplicates_block = content_blocks_element("duplicates_analysis",
		duplicates_analysis_data)

	# block analysis
	n_blocks, largest_block, n_ungrouped = preprocessing._info_blocks_stats
	block_analysis_data = {
		"index": ["information blocks", "column largest block",
			"ungrouped columns"],
		"data": [n_blocks, largest_block, n_ungrouped]
	}

	block_analysis_block = content_blocks_element("block_analysis",
		block_analysis_data)

	# cpu time
	cpu_time_data = {
		"index": ["total", "categories", "duplicates",
			"information blocks"],
		"data": [preprocessing._time_run, preprocessing._time_cases,
			preprocessing._time_duplicates, preprocessing._time_info_blocks]
	}

	cpu_time_block = content_blocks_element("cpu_time", cpu_time_data)

	# build stats step contents
	content_blocks = [db_info_block	, column_analysis_block,
		duplicates_block, block_analysis_block, cpu_time_block]

	step_contents_stats_run = step_contents_element(
		content_position="sidebar_left", content_type="stats",
		content_blocks=content_blocks)

	# preprocessing transform statistics
	if preprocessing._is_transformed:
		param_data = {
			"index": ["mode"],
			"data": [preprocessing._transform_mode]
		}

		param_block = content_blocks_element("parameters", param_data)
		# build param step contents
		content_blocks = [param_block]
		step_contents_config_transform = step_contents_element(
			content_position="sidebar_left", content_type="parameters",
			content_blocks=content_blocks)

		# preprocessing transform statistics
		n_columns_transform = sum([d["status"] in ["nan_unique",
			"numeric_transform"] for d in dict_values])

		n_columns_after = preprocessing._n_columns - preprocessing._n_columns_remove

		n_user_actions = sum(1 for d in dict_values if "user_action" in d)
		n_user_comment = sum(d["user_comment"] != "" for d in dict_values
			if "user_action" in d)

		results_data = {
			"index": [
				"original data", "after preprocessing", "removed",
				"transformed", "user actions", "user_comments"
				],
			"data": [preprocessing._n_columns, n_columns_after,
				preprocessing._n_columns_remove, n_columns_transform,
				n_user_actions, n_user_comment]
		}

		results_block = content_blocks_element("results", results_data)

		# cpu time
		cpu_time_data = {
			"index": ["total", "remove", "transform"],
			"data": [preprocessing._time_transform, preprocessing._time_remove,
				preprocessing._time_apply_transform]
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
		"grmlabcls": preprocessing,
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


def add_preprocessing_data_stream(preprocessing,
	item_title="Data Preprocessing"):
	# preprocessing parameters
	if preprocessing.target is None:
		target = NOT_AVAILABLE
	else:
		target = preprocessing.target

	if preprocessing.date is None:
		date = NOT_AVAILABLE
	else:
		date = preprocessing.date

	if not preprocessing.special_values:
		special_values = NOT_AVAILABLE
	else:
		special_values = AVAILABLE

	verbose = preprocessing.verbose

	param_data = {
		"index": ["target", "date", "special values", "verbose"],
		"data": [target, date, special_values, verbose]
	}

	param_block = content_blocks_element("parameters", param_data)
	# build param step contents
	content_blocks = [param_block]
	step_contents_config_run = step_contents_element(
		content_position="sidebar_left", content_type="parameters",
		content_blocks=content_blocks)

	# preprocessing run statistics
	db_info_data = {
		"index": ["samples", "columns", "file size (GB)", "chunks"],
		"data": [preprocessing._n_samples, preprocessing._n_columns,
			preprocessing._file_size, preprocessing._n_chunks]
	}

	db_info_block = content_blocks_element("db_info", db_info_data)

	# column analysis
	dict_values = preprocessing._dict_column.values()

	categories = ["binary", "constant", "date", "empty", "exclude", "id",
			"nan_unique", "numeric_conversion", "ok", "special",
			"special_constant", "special_unique"]

	categories_data = [sum([d["status"] == c for d in dict_values])
		for c in categories]

	column_analysis_data = {
		"index": categories,
		"data": categories_data
	}

	column_analysis_block = content_blocks_element("column_analysis",
		column_analysis_data)

	# duplicates analysis
	n_duplicates = sum([d["duplicate"] for d in dict_values])
	n_duplicates_of = sum([1 if d["duplicate_of"] is not None else 0
		for d in dict_values])

	duplicates_analysis_data = {
		"index": ["duplicates", "duplicates_of"],
		"data": [n_duplicates, n_duplicates_of]
	}

	duplicates_block = content_blocks_element("duplicates_analysis",
		duplicates_analysis_data)

	# block analysis
	n_blocks, largest_block, n_ungrouped = preprocessing._info_blocks_stats
	block_analysis_data = {
		"index": ["information blocks", "column largest block",
			"ungrouped columns"],
		"data": [n_blocks, largest_block, n_ungrouped]
	}

	block_analysis_block = content_blocks_element("block_analysis",
		block_analysis_data)

	# cpu time
	cpu_time_data = {
		"index": ["total", "map", "reduce", "io"],
		"data": [preprocessing._time_run, preprocessing._time_map,
			preprocessing._time_reduce, preprocessing._time_io]
	}

	cpu_time_block = content_blocks_element("cpu_time", cpu_time_data)

	# build stats step contents
	content_blocks = [db_info_block	, column_analysis_block,
		duplicates_block, block_analysis_block, cpu_time_block]

	step_contents_stats_run = step_contents_element(
		content_position="sidebar_left", content_type="stats",
		content_blocks=content_blocks)

	item_info = {
		"grmlabcls": preprocessing,
		"item_title": item_title,
		"step_config_run": step_contents_config_run,
		"step_config_transform": None,
		"step_stats_run": step_contents_stats_run,
		"step_stats_transform": None,
		"results_run_names": None,
		"results_transform_names": None,
		"extra_item_step": None
	}

	return item_info
