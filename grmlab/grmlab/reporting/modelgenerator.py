"""
Model generator class reporting.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

from .util import content_blocks_element
from .util import step_contents_element


# reporting messages
AVAILABLE = "available"
NOT_AVAILABLE = "not available"


def add_model_generator(modelgenerator, item_title="Model Generator"):
	# modelgenerator parameters
	if modelgenerator.feature_names is None:
		feature_names = NOT_AVAILABLE
	else:
		feature_names = modelgenerator.feature_names

	if modelgenerator.feature_selection is None:
		feature_selection = NOT_AVAILABLE
	else:
		feature_selection = AVAILABLE

	if modelgenerator.estimator is None:
		estimator = NOT_AVAILABLE
	else:
		estimator = AVAILABLE

	test_size = modelgenerator.test_size
	random_state = modelgenerator.random_state
	n_jobs = modelgenerator.n_jobs
	verbose = modelgenerator.verbose

	param_data = {
		"index": ["feature names", "feature selection", "estimator",
			"test_size", "random_state", "n_jobs", "verbose"],
		"data": [feature_names, feature_selection, estimator, test_size,
			random_state, n_jobs, verbose]
	}

	param_block = content_blocks_element("parameters", param_data)
	# build param step contents
	content_blocks = [param_block]
	step_contents_config_run = step_contents_element(
		content_position="sidebar_left", content_type="parameters",
		content_blocks=content_blocks)	

	# model generator run statistics
	db_info_data = {
		"index": ["samples", "variables", "target dtype"],
		"data": [modelgenerator._n_samples, modelgenerator._n_features,
			modelgenerator._target_dtype]
	}

	db_info_block = content_blocks_element("db_info_expanded", db_info_data)

	# models analysis
	model_analysis_data = {
		"index": ["samples train set", "samples test set", "models",
			"feature selection", "estimators"],
		"data": [modelgenerator._n_samples_train,
			modelgenerator._n_samples_test, modelgenerator.n_models_,
			modelgenerator._n_feature_selection, modelgenerator._n_estimator]
	}

	model_analysis_block = content_blocks_element("model_analysis",
		model_analysis_data)

	# cpu time
	time_feature_selection = sum(model["time_feature_selection"] for model in
		modelgenerator._models_metrics)

	time_estimator = sum(model["time_estimator"] for model in
		modelgenerator._models_metrics)

	time_others = modelgenerator._time_run - (time_feature_selection + time_estimator)

	cpu_time_data = {
		"index": ["total", "feature selection", "estimators", "others"],
		"data": [modelgenerator._time_run, time_feature_selection,
			time_estimator, time_others]
	}

	cpu_time_block = content_blocks_element("cpu_time", cpu_time_data)

	# build stats step contents
	content_blocks = [db_info_block, model_analysis_block, cpu_time_block]

	step_contents_stats_run = step_contents_element(
		content_position="sidebar_left", content_type="stats",
		content_blocks=content_blocks)

	item_info = {
		"grmlabcls": modelgenerator,
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