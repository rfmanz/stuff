"""
Model optimizer class reporting.
"""

# Authors: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2019.

import numpy as np

from .util import content_blocks_element
from .util import dict_to_json
from .util import step_contents_element


# JSON code
JSON_DATA_ITEMS = "json_items_data['{}'] = ('{}');"


def add_model_optimizer(info, item_id):
	# variable information - run step ("01")
	step_id = "01"
	content_position = 2
	block_position = 0

	# input model
	input_model = {
		"characteristics": {
			"index": ["name", "target_type", "estimator", "problem_class"], 
			"data": [info.model.name, 
					 info._target_dtype, 
					 type(info.model.estimator).__name__,
					 info.model.feature_selection._solver._mipcl_problem.name]
		},
		"feature_selection":{
			"index": ["n_min_features", "n_max_features", "max_correlation", 
					  "abs_correlation", "excluded_variables", 
					  "forced_variables",  "selected_variables", 
					  "infeasible_pairs", "group_constraints"],
			"data": [info.model.feature_selection.n_min_features,
					 info.model.feature_selection.n_max_features,
					 info.model.feature_selection.max_correlation,
					 str(info.model.feature_selection.abs_correlation),
					 info.model.feature_selection._solver._n_excluded,
					 info.model.feature_selection._solver._n_fixed,
					 info.model.feature_selection._solver._n_selected_features, 
					 info.model.feature_selection._solver._n_infeas_pairs,
					 info.model.feature_selection._solver._n_groups]
		},
		"estimator": {
			"index": list(info.model.get_params_estimator().keys()),
			"data": [(str(elt) if isinstance(elt, bool) else elt
						) if elt is not None else "nan" 
						for elt in info.model.get_params_estimator().values()]
		}
	}

	# train dataset
	train_input =  {
		"index": ["samples", "columns"],
		"data": [info.model._n_samples, 
				 info.model._n_features]
	}

	# scorer
	scorer = {
		"index":info.scorer.metrics,
		"data":info.scorer.weights
	}

	# params
	param_dic = {
		"index":["name", "class", "optimal", "distribution",  "min", "max", "choice", "iters"], 
		"data":[]
	}
	for key in info.parameters._dict_params:
	    split_key = key.split("__")
	    param_content = info.parameters._dict_params[key]
	    # statement to detect if the parameter values have been set with a range
	    # with min and max value or with a list.
	    if isinstance(param_content[-1],(list,np.ndarray)):
	    	# Parameters with choice have the min and max values set as None
	        param_dic["data"].append([
	        	split_key[1],split_key[0], 
				info.best_model_parameters[key],
				param_content[0], 
				None, None,
				param_content[1],
				[obj["params"][key] for obj in info._trials.results]
			])
	    else:
	    	# Parameters with min and max have the choice value set as None
	        param_dic["data"].append([
				split_key[1],split_key[0], 
				info.best_model_parameters[key], 
				param_content[0], 
				param_content[1], param_content[2], None,
				[obj["params"][key] for obj in info._trials.results]
			])


	# best model
	best_model = {
		"train_input": {
			"index": ["samples", "columns"],
			"data": [info.best_model._n_samples, 
					 info.best_model._n_features]
		},
		"characteristics": {
			"index": ["name", "target_type", "estimator", "problem_class"], 
			"data": [info.best_model.name, 
					 info._target_dtype, 
					 type(info.best_model.estimator).__name__,
					 info.best_model.feature_selection._solver._mipcl_problem.name]
		},
		"feature_selection":{
			"index": ["n_min_features", "n_max_features", "max_correlation", 
					  "abs_correlation", "excluded_variables", 
					  "forced_variables",  "selected_variables", 
					  "infeasible_pairs", "group_constraints"],
			"data": [info.best_model.feature_selection.n_min_features,
					 info.best_model.feature_selection.n_max_features,
					 info.best_model.feature_selection.max_correlation,
					 str(info.best_model.feature_selection.abs_correlation),
					 info.best_model.feature_selection._solver._n_excluded,
					 info.best_model.feature_selection._solver._n_fixed,
					 info.best_model.feature_selection._solver._n_selected_features, 
					 info.best_model.feature_selection._solver._n_infeas_pairs,
					 info.best_model.feature_selection._solver._n_groups]
		},
		"estimator": {
			"index": list(info.best_model.get_params_estimator().keys()),
			"data": [(str(elt) if isinstance(elt, bool) else elt
						) if elt is not None else "nan" 
						for elt in info.best_model.get_params_estimator().values()]
		}
	}
	# iterations info
	iter_vec = {
		"index": ['features', 'coefs', 'score_max', 'score_mean', 'score_min', 'score_std', 'loss'], 
		"data": []
	}
	for obj in info._trials.results:
		info_iter = []
		info_iter += [obj['feature_names'], obj['feature_coef'],
					  obj['score_max'],obj['score_mean'],
					  obj['score_min'],obj['score_std']]
		info_iter += [obj['loss']]
		iter_vec["data"].append(info_iter)

	# all info of model optimizer
	optimizer_metrics = {
		"characteristics":{
			"index": ["algorithm", "n_iters", "n_split"],
			"data": [info.algorithm, info.n_iters, info.n_splits]
		},
		"cpu_time":{
			"index": ["total"],
			"data": [info._time_solver]
		},
		"input_model": input_model,
		"train_input" : train_input,
		"scorer": scorer,
		"parameters": param_dic,
		"iters_info": iter_vec,
		"output_model": best_model
	}

	# variable structure
	json_optimizer = {
		"item_id": item_id,
		"step_id": step_id,
		"content_position": content_position,
		"block_position": block_position,
		"data": optimizer_metrics
	}

	optimizer_data = content_blocks_element("optimizer_data", json_optimizer)
	step_contents = step_contents_element(
		content_position="", content_type="",
		content_blocks=[optimizer_data])

	return step_contents