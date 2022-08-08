"""
Model Comparison class reporting.
"""

# Authors: Carlos González Berrendero <c.gonzalez.berrender@bbva.com>
# BBVA - Copyright 2019.

import numpy as np

from .util import content_blocks_element
from .util import dict_to_json
from .util import step_contents_element


def add_model_comparison(info, item_id):
	# variable information - run step ("01")
	step_id = "01"
	content_position = 2
	block_position = 0

	# Prediction
	agree_correct_pred = (info._agreement_predictions_sum[0] +
					info._agreement_predictions_sum[3])
	agree_incorrect_pred = (info._agreement_predictions_sum[1] +
					info._agreement_predictions_sum[2])
	agree_pred = agree_correct_pred + agree_incorrect_pred

	# Features
	all_features = info._common_features.copy()
	maxs = [max(var[1], var[2]) for var in all_features]

	for var in info._only_in_1_features:
		var_append = (*var, '*')
		maxs.append(var[1])
		all_features.append(var_append)
	for var in info._only_in_2_features:
		var_append = (var[0], '*', var[1])
		maxs.append(var[1])
		all_features.append(var_append)
	ind_sort = np.argsort(maxs)
	all_features = np.array(all_features)
	all_features = all_features[ind_sort[::-1]]
	len_max = max([len(x[0]) for x in all_features])
	for var in all_features:
		var[1] = var[1][0:6]
		var[2] = var[2][0:6]

	feature_analysis = {'info': ['name', 'Imp. Model 1', 
								'Imp. Model 2', 'Coef. sign'],
						'data': []}

	for feature in info._common_features:
		name = feature[0]
		imp_m1 = feature[1]
		imp_m2 = feature[2]
		try:
			coef_m1 = info._analyzer1._feature_coefficients[name]
			coef_m2 = info._analyzer2._feature_coefficients[name]
			if coef_m1 * coef_m2 < 0:
				coef_sign = 'WARN'
			else:
				coef_sign = 'OK'
		except: 
			coef_sign = ''
		feature_analysis['data'].append([name, imp_m1, imp_m2, coef_sign])

	for feature in info._only_in_1_features:
		feature_analysis['data'].append([feature[0], feature[1], '', ''])

	for feature in info._only_in_2_features:
		feature_analysis['data'].append([feature[0], '', feature[1], ''])

	# metrics
	# Same analysis for same metrics
	metrics_random = []
	metrics_temporal = []
	for name in info._analyzer1._dict_metrics:
		if "values_rand" in info._analyzer1._dict_metrics[name].keys():
			metrics_random.append(name)
		if "temporal" in info._analyzer1._dict_metrics[name].keys():
			metrics_temporal.append(name)
			
	temp_vars = {"count_temporal": info._analyzer1._count_temporal,
				"intervals_temporal": info._analyzer1._intervals_temporal,
				"metrics_random": metrics_random,
				"metrics_temporal": metrics_temporal,
				}

	model_characteristics = {
		"train_input": {
			"Model 1":{ "index": ["samples", "columns"],
						"data": [info._analyzer1._n_samples, 
					 	info._analyzer1._n_columns]},
			"Model 2":{ "index": ["samples", "columns"],
						"data": [info._analyzer2._n_samples, 
					 	info._analyzer2._n_columns]}
		},
		"characteristics": {
			"Model 1":{	"index": ["name", "target_type", "estimator"], 
						"data": [info._analyzer1._model.name, 
								 None,#info._analyzer1._model.feature_selection._solver._target_type, 
								 type(info._analyzer1._model.estimator).__name__]},
			"Model 2":{"index": ["name", "target_type", "estimator"], 
						"data": [info._analyzer2._model.name, 
								 None,#info._analyzer2._model.feature_selection._solver._target_type, 
								 type(info._analyzer2._model.estimator).__name__]}	 
		},
		"feature_selection":{
			"Model 1": {"index": ["min_features", "max_features", "max_correlation", 
					  "abs_correlation", "excluded_variables", 
					  "forced_variables",  "selected_variables", 
					  "infeasible_pairs", "group_constraints"],
				"data": [info._analyzer1._model.feature_selection._solver.n_min_features,
						 info._analyzer1._model.feature_selection._solver.n_max_features,
						 info._analyzer1._model.feature_selection.max_correlation,
						 str(info._analyzer1._model.feature_selection.abs_correlation),
						 info._analyzer1._model.feature_selection._solver._n_excluded,
						 info._analyzer1._model.feature_selection._solver._n_fixed,
						 info._analyzer1._model.feature_selection._solver._n_selected_features, 
						 info._analyzer1._model.feature_selection._solver._n_infeas_pairs,
						 info._analyzer1._model.feature_selection._solver._n_groups]},
			"Model 2": {"index": ["min_features", "max_features", "max_correlation", 
					  "abs_correlation", "excluded_variables", 
					  "forced_variables",  "selected_variables", 
					  "infeasible_pairs", "group_constraints"],
				"data": [info._analyzer2._model.feature_selection._solver.n_min_features,
						 info._analyzer2._model.feature_selection._solver.n_max_features,
						 info._analyzer2._model.feature_selection.max_correlation,
						 str(info._analyzer2._model.feature_selection.abs_correlation),
						 info._analyzer2._model.feature_selection._solver._n_excluded,
						 info._analyzer2._model.feature_selection._solver._n_fixed,
						 info._analyzer2._model.feature_selection._solver._n_selected_features, 
						 info._analyzer2._model.feature_selection._solver._n_infeas_pairs,
						 info._analyzer2._model.feature_selection._solver._n_groups]}
		},
		"estimator": {
			"Model 1":{ "index": list(info._analyzer1._model.get_params_estimator().keys()),
					"data": [(str(x) if isinstance(x, bool) else x
						) if x is not None else "nan" 
						for x in info._analyzer1._model.get_params_estimator().values()]},
			"Model 2":{ "index": list(info._analyzer2._model.get_params_estimator().keys()),
					"data": [(str(x) if isinstance(x, bool) else x
						) if x is not None else "nan" 
						for x in info._analyzer2._model.get_params_estimator().values()]}
		}
	}

	# all info of model comparison inputs and characteristics
	roc_fpr = np.linspace(0, 1, 101)
	roc_tpr_m1 = np.interp(roc_fpr, info._analyzer1._roc[0], info._analyzer1._roc[1])
	roc_tpr_m2 = np.interp(roc_fpr, info._analyzer2._roc[0], info._analyzer2._roc[1])

	comparison_data = {
		"prediction":{
			"index":["n_samples", "agree_pred", "agree_pred_perct",
					"agree_correct_pred", "agree_correct_pred_perct",
					"agree_incorrect_pred", "agree_incorrect_pred_perct",
					"l2_dif_pred", "disagree_v1_1_0", "disagree_v1_0_1",
					"disagree_v0_1_0", "disagree_v0_0_1"],
			"data":[info._analyzer1._n_samples, agree_pred,
			agree_pred/info._analyzer1._n_samples*100, agree_correct_pred,
			agree_correct_pred/info._analyzer1._n_samples*100,
			agree_incorrect_pred,
			agree_incorrect_pred/info._analyzer1._n_samples*100,
			info._l2_dif_pred,info._disagreement_predictions_sum[0],
			info._disagreement_predictions_sum[1],
			info._disagreement_predictions_sum[2],
			info._disagreement_predictions_sum[3]]
		},
		"roc": {
			"fpr": roc_fpr,
			"tpr_m1": roc_tpr_m1,
			"tpr_m2": roc_tpr_m2
		},
		"metrics":{
			"Model 1": info._analyzer1._dict_metrics,
			"Model 2": info._analyzer2._dict_metrics
		},
		"features":{
			"l2_dif_feature":info._l2_dif_feature,
			"index":["name", "imp_m1", "imp_m2"],
			"data": all_features,
		},
		"characteristics":{
			"index": ["name_comp","name_m1","name_m2"],
			"data": [info.name,info._analyzer1.name,info._analyzer2.name]
		},
		"test_input":{
			"index": ["samples", "columns"],
			"data": [info._analyzer1._n_samples, info._analyzer1._n_columns]},
		"run": {
			"index":["n_simulations", "simulation_size", "approximations", 
			"Time analysis"],
			"data": [info.n_simulations, info.simulation_size, 
					 str(info.feature_analysis_approximation), 
					 str(info._is_dates_provided)]
		},
		"features": feature_analysis,
		"model_characteristics": model_characteristics,
		**temp_vars
	}

	# variable structure
	json_comparison = {
		"item_id": item_id,
		"step_id": step_id,
		"content_position": content_position,
		"block_position": block_position,
		"data": comparison_data
	}

	analysis_data = content_blocks_element("comparison_data", json_comparison)
	step_contents = step_contents_element(
		content_position="", content_type="",
		content_blocks=[analysis_data])

	return step_contents
