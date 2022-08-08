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


def add_model_analysis(info, item_id):
	# variable information - run step ("01")
	step_id = "01"
	content_position = 2
	block_position = 0

	# features
	feature_analysis = dict()
	feature_analysis["importance"] = {"info": ["name", "coeff.","importance", "vif"],
									  "data": []}
	for i, (name, value) in enumerate(info._feature_importance):
		coef = info._feature_coefficients[name] if (
			info._feature_coefficients is not None) else None
		feature_analysis["importance"]["data"].append([
			name, coef, value, info._vif[name]])
	feature_analysis["correlations"] = {
		"info": list(info._feature_names), 
		"data": [[round(elt, 8) for elt in vec] for vec in info._correlations],
		"statistics": info._correlation_metrics
		}

	# metrics
	metrics_random = []
	metrics_temporal = []
	for name in info._dict_metrics:
		if "values_rand" in info._dict_metrics[name].keys():
			metrics_random.append(name)
		if "temporal" in info._dict_metrics[name].keys():
			metrics_temporal.append(name)
	temp_vars = {"count_temporal": info._count_temporal, 
				 "intervals_temporal": info._intervals_temporal, 
				 "metrics_random": metrics_random,
				 "metrics_temporal": metrics_temporal}

	if info._model._is_feature_selection_fitted:
		feat_sel_params = {
			"index": ["min_features", "max_features", "max_correlation", 
					  "abs_correlation", "excluded_variables", 
					  "forced_variables",  "selected_variables", 
					  "infeasible_pairs", "group_constraints"],
			"data": [info._model.feature_selection.n_min_features,
					 info._model.feature_selection.n_max_features,
					 info._model.feature_selection.max_correlation,
					 str(info._model.feature_selection.abs_correlation),
					 info._model.feature_selection._solver._n_excluded,
					 info._model.feature_selection._solver._n_fixed,
					 info._model.feature_selection._solver._n_selected_features, 
					 info._model.feature_selection._solver._n_infeas_pairs,
					 info._model.feature_selection._solver._n_groups]
		}
		feat_sel_problem_name = info._model.feature_selection._solver._mipcl_problem.name
	else:
		feat_sel_params =  {
			"index": ["min_features", "max_features", "max_correlation", 
					  "abs_correlation", "excluded_variables", 
					  "forced_variables",  "selected_variables", 
					  "infeasible_pairs", "group_constraints"],
			"data": ["", "", "", "", "", "", "", "", ""]
		}
		feat_sel_problem_name = ""

	# model characteristics
	model_characteristics = {
		"train_input": {
			"index": ["samples", "columns"],
			"data": [info._model._n_samples, 
					 info._model._n_features]
		},
		"characteristics": {
			"index": ["name", "target_type", "estimator", "problem_class"], 
			"data": [info._model.name, 
					 "None",#info._model.feature_selection._solver._target_type, 
					 type(info._model.estimator).__name__,
					 feat_sel_problem_name]
		},
		"feature_selection": feat_sel_params,
		"estimator": {
			"index": list(info._model.get_params_estimator().keys()),
			"data": [(str(elt) if isinstance(elt, bool) else elt
						) if elt is not None else "nan" 
						for elt in info._model.get_params_estimator().values()]
		}
	}

	# all info of model analysis inputs and characteristics
	analyzer_metrics = {
		"characteristics":{
			"index": ["name"],
			"data": [info.name]
		},
		"test_input":{
			"index": ["samples", "columns"],
			"data": [info._n_samples, info._n_columns]},
		"run": {
			"index":["n_simulations", "simulation_size", "approximations", 
			"Time analysis"],
			"data": [info.n_simulations, info.simulation_size, 
					 str(info.feature_analysis_approximation), 
					 str(info._is_dates_provided)]
		},
		"roc": {
			"fpr": [elt for i, elt in enumerate(info._roc[0]) if i % int(
				len(info._roc[0])/100 + 1) == 0], 
			"tpr": [elt for i, elt in enumerate(info._roc[1]) if i % int(
				len(info._roc[1])/100 + 1) == 0]
		},
		"cpu_time":{
			"index": ["total","metrics", "feature analysis"],
			"data": [info._time_run, info._time_run_metrics, 
					 info._time_run_feature_analysis]
		},
		"metrics": info._dict_metrics,
		"features": feature_analysis,
		"model_characteristics": model_characteristics,
		**temp_vars
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


def add_model_analysis_continuous(info, item_id):
	# variable information - run step ("01")
	step_id = "01"
	content_position = 2
	block_position = 0

	# features
	feature_analysis = dict()
	feature_analysis["importance"] = {"info": ["name", "coeff.","importance", "vif"],
									  "data": []}
	for i, (name, value) in enumerate(info._feature_importance):
		coef = info._feature_coefficients[name] if (
			info._feature_coefficients is not None) else None
		feature_analysis["importance"]["data"].append([
			name, coef, value, info._vif[name]])
	feature_analysis["correlations"] = {
		"info": list(info._feature_names), 
		"data": [[round(elt, 8) for elt in vec] for vec in info._correlations],
		"statistics": info._correlation_metrics
		}

	# confusion Matrix
	conf_mat = {
		"info_pred": list(info.conf_mat_names[0]),
		"info_true": list(info.conf_mat_names[1]),
		"data": [[elt for elt in vec] for vec in info.conf_mat]}
	# error Matrix
	error_mat = {
		"info_true": list(info.error_mat_names[0]),
		"info_error": list(info.error_mat_names[1]),
		"data": [[elt for elt in vec] for vec in info.error_mat]}

	# metrics
	metrics_random = []
	metrics_temporal = []
	for name in info._dict_metrics:
		if "values_rand" in info._dict_metrics[name].keys():
			metrics_random.append(name)
		if "temporal" in info._dict_metrics[name].keys():
			metrics_temporal.append(name)
	temp_vars = {"count_temporal": info._count_temporal, 
				 "intervals_temporal": info._intervals_temporal, 
				 "metrics_random": metrics_random,
				 "metrics_temporal": metrics_temporal}

	if info._model._is_feature_selection_fitted:
		feat_sel_params = {
			"index": ["min_features", "max_features", "max_correlation", 
					  "abs_correlation", "excluded_variables", 
					  "forced_variables",  "selected_variables", 
					  "infeasible_pairs", "group_constraints"],
			"data": [info._model.feature_selection.n_min_features,
					 info._model.feature_selection.n_max_features,
					 info._model.feature_selection.max_correlation,
					 str(info._model.feature_selection.abs_correlation),
					 info._model.feature_selection._solver._n_excluded,
					 info._model.feature_selection._solver._n_fixed,
					 info._model.feature_selection._solver._n_selected_features, 
					 info._model.feature_selection._solver._n_infeas_pairs,
					 info._model.feature_selection._solver._n_groups]
		}
		feat_sel_problem_name = info._model.feature_selection._solver._mipcl_problem.name
	else:
		feat_sel_params =  {
			"index": ["min_features", "max_features", "max_correlation", 
					  "abs_correlation", "excluded_variables", 
					  "forced_variables",  "selected_variables", 
					  "infeasible_pairs", "group_constraints"],
			"data": ["", "", "", "", "", "", "", "", ""]
		}
		feat_sel_problem_name = ""

	# model characteristics
	model_characteristics = {
		"train_input": {
			"index": ["samples", "columns"],
			"data": [info._model._n_samples, 
					 info._model._n_features]
		},
		"characteristics": {
			"index": ["name", "target_type", "estimator", "problem_class"], 
			"data": [info._model.name, 
					 "None",#info._model.feature_selection._solver._target_type, 
					 type(info._model.estimator).__name__,
					 feat_sel_problem_name]
		},
		"feature_selection": feat_sel_params,
		"estimator": {
			"index": list(info._model.get_params_estimator().keys()),
			"data": [(str(elt) if isinstance(elt, bool) else elt
						) if elt is not None else "nan" 
						for elt in info._model.get_params_estimator().values()]
		}
	}

	# all info of model analysis inputs and characteristics
	analyzer_metrics = {
		"characteristics":{
			"index": ["name"],
			"data": [info.name]
		},
		"test_input":{
			"index": ["samples", "columns"],
			"data": [info._n_samples, info._n_columns]},
		"run": {
			"index":["n_simulations", "simulation_size", "approximations", 
			"Time analysis"],
			"data": [info.n_simulations, info.simulation_size, 
					 str(info.feature_analysis_approximation), 
					 str(info._is_dates_provided)]
		},
		"cpu_time":{
			"index": ["total","metrics", "feature analysis"],
			"data": [info._time_run, info._time_run_metrics, 
					 info._time_run_feature_analysis]
		},
		"metrics": info._dict_metrics,
		"features": feature_analysis,
		"confusion_matrix": conf_mat,
		"error_matrix": error_mat,
		"model_characteristics": model_characteristics,
		**temp_vars
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