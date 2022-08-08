"""
Core GRMlab reporting functions.

BBVA - GRM Team Copyright 2018.
"""
import json
import os
import platform
import shutil
import sys

from grmlab.data_analysis.univariate import Univariate
from grmlab.data_analysis.bivariate import Bivariate
from grmlab.data_processing.feature_binning.optimalgrouping import OptimalGrouping
from grmlab.data_processing.preprocessing.basic import Preprocessing
from grmlab.modelling.model_generator.generator import ModelGenerator
from .util import dict_to_json, get_id, is_report, reporting_class_type_check
from .util import user_info

# JSON code
_JSON_CONTENTS_USER_INFO = "var user_info = '{}';"
_JSON_CONTENTS_REPORT_INFO = "var report_info = '{}';"
_JSON_CONTENTS_ITEMS_ARRAY = "var json_items = new Array();"
_JSON_CONTENTS_ITEMS = "json_items.push('{}');"
_JSON_DATA_ITEMS_OBJECT = "var json_items_data = new Object();"
_JSON_DATA_ITEMS = "json_items_data['{}'] = ('{}');"


class Reporting(object):
	"""docstring for Reporting"""
	def __init__(self, path, verbose=False, **options):
		self._path = path

		# optional arguments
		self._report_title = options.get("title", "")
		self._report_description = options.get("description", "")
		self._report_date = options.get("date", "")

		# main json files
		self._json_contents_file = None
		self._json_data_file = None

		# initialize report
		self._init_report()

	def add_preprocessing(self, preprocessing, item_title="Data Preprocessing"):

		# check class type
		reporting_class_type_check("preprocessing", preprocessing, Preprocessing)

		# preprocessing config options
		# ======================================================================
		config_data = {
			"index": [
				"special values"
				],
			"data": [
				"available" if preprocessing._special_values else "not available"
				]
		}

		config_block = self._content_blocks_element("config", config_data)

		# build config step contents
		content_blocks = [config_block]
		step_contents_config = self._step_contents_element(
			content_position="sidebar_left", content_type="config", 
			content_blocks=content_blocks)

		# preprocessing analysis statistics
		# ======================================================================
		# db info
		db_info_data = {
			"index": [
				"records", 
				"columns"
				],								
			"data": [
				preprocessing._n, 
				preprocessing._nvars
				]
		}

		db_info_block = self._content_blocks_element("db_info", db_info_data)

		# column analysis
		column_analysis_data = {
			"index": [
				"empty", 
				"constant", 
				"id", 
				"dates", 
				"binary", 
				"target", 
				"NaN-unique", 
				"NaN-special", 
				"numeric_conversion", 
				"special", 
				"duplicated"
				],
			"data": [
				preprocessing._nvars_empty,
				preprocessing._nvars_constant,
				preprocessing._nvars_id,
				preprocessing._nvars_dates,
				preprocessing._nvars_binary,
				preprocessing._nvars_target,
				preprocessing._nvars_nan_unique,
				preprocessing._nvars_special_unique,
				preprocessing._nvars_numeric_conv,
				preprocessing._nvars_special,
				preprocessing._nvars_duplicated
				]
		}

		column_analysis_block = self._content_blocks_element("column_analysis", 
			column_analysis_data)

		# block analysis
		block_analysis_data = {
			"index": [
				"information blocks",
				"column largest block",
				"ungrouped columns"
				],
			"data": [
				preprocessing._n_blocks,
				preprocessing._n_largest_block,
				preprocessing._n_ungrouped_blocks
				]
		}

		block_analysis_block = self._content_blocks_element("block_analysis", 
			block_analysis_data)
			
		# cpu time
		cpu_time_data = {
			"index": [
				"total",
				"classification",
				"duplicates",
				"information blocks"
				],
			"data": [
				preprocessing._time_analysis_total,
				preprocessing._time_analysis_classification,
				preprocessing._time_analysis_duplicates,
				preprocessing._time_analysis_info_blocks
				]
		}

		cpu_time_block = self._content_blocks_element("cpu_time", cpu_time_data)

		# build stats step contents
		content_blocks = [db_info_block	, column_analysis_block, 
			block_analysis_block, cpu_time_block]
		step_contents_stats_analysis = self._step_contents_element(
			content_position="sidebar_left", content_type="stats", 
			content_blocks=content_blocks)

		# preprocessing apply statistics
		# ======================================================================
		# results
		results_data = {
			"index": [
				"original data",
				"after preprocessing",
				"removed",
				"transformed"
				],
			"data": [
				preprocessing._nvars,
				preprocessing._nvars_after_preprocessing,
				preprocessing._nvars_removed,
				preprocessing._nvars_transformed
				]
		}

		results_block = self._content_blocks_element("results", results_data)

		# cpu time
		cpu_time_data = {
			"index": [
				"total",
				"remove",
				"transform"
				],
			"data": [
				preprocessing._time_apply_total,
				preprocessing._time_apply_remove,
				preprocessing._time_apply_transform
				]
		}

		cpu_time_block = self._content_blocks_element("cpu_time", cpu_time_data)

		content_blocks = [results_block, cpu_time_block]
		step_contents_stats_apply = self._step_contents_element(
			content_position="sidebar_left", content_type="stats", 
			content_blocks=content_blocks)

		# build item
		self._build_item(grmlabcls=preprocessing, item_title=item_title,
			step_config=step_contents_config,
			step_stats_analysis=step_contents_stats_analysis,
			step_stats_apply=step_contents_stats_apply)

	def add_univariate(self, univariate, item_title="Univariate Analysis"):

		# check class type
		reporting_class_type_check("univariate", univariate, Univariate)

		# univariate config options
		# ======================================================================
		config_data = {
			"index": [
				"special values",
				"nominal variables"
				],
			"data": [
				"available" if univariate._special_values else "not available",
				"available" if univariate._variables_nominal else "not available"
				]
		}

		config_block = self._content_blocks_element("config", config_data)

		# build config step contents
		content_blocks = [config_block]
		step_contents_config = self._step_contents_element(
			content_position="sidebar_left", content_type="config", 
			content_blocks=content_blocks)

		# univariate analysis statistics
		# ======================================================================
		# db info
		db_info_data = {
			"index": [
				"records", 
				"columns",
				"date column",
				"target column",
				"target type"
				],			
			"data": [
				univariate._n, 
				univariate._nvars,
				univariate._date_name,
				univariate._target_name,
				univariate._target_type
				]
		}

		db_info_block = self._content_blocks_element("db_info_expanded", 
			db_info_data)

		# column analysis data
		column_analysis_data = {
			"index": [
				"numerical",
				"ordinal",
				"categorical",
				"nominal",
				"high missing",
				"high special",
				"high stability info",
				"high stability values"
				],
			"data": [
				univariate._nvars_numerical,
				univariate._nvars_ordinal,
				univariate._nvars_categorical,
				univariate._nvars_nominal,
				univariate._nvars_missing,
				univariate._nvars_special,
				univariate._nvars_stability_info,
				univariate._nvars_stability_values
				]
		}

		column_analysis_block = self._content_blocks_element("column_analysis", 
			column_analysis_data)

		# cpu time
		cpu_time_data = {
			"index": [
				"total",
				"numerical",
				"ordinal",
				"categorical",
				"nominal"
				],
			"data": [
				univariate._time_analysis_total,
				univariate._time_analysis_numerical,
				univariate._time_analysis_ordinal,
				univariate._time_analysis_categorical,
				univariate._time_analysis_nominal
				]
		}

		cpu_time_block = self._content_blocks_element("cpu_time", cpu_time_data)

		content_blocks = [db_info_block, column_analysis_block, cpu_time_block]
		step_contents_stats_analysis = self._step_contents_element(
			content_position="sidebar_left", content_type="stats", 
			content_blocks=content_blocks)

		# univariate apply statistics
		# ======================================================================
		# results
		results_data = {
			"index": [
				"original data",
				"after univariate",
				"removed",
				],
			"data": [
				univariate._nvars,
				univariate._nvars_after_apply,
				univariate._nvars_removed,
				]
		}

		results_block = self._content_blocks_element("results", results_data)

		# cpu time
		cpu_time_data = {
			"index": [
				"total",
				"remove"
				],
			"data": [
				univariate._time_apply_total,
				univariate._time_apply_remove
				]
		}

		cpu_time_block = self._content_blocks_element("cpu_time", cpu_time_data)

		content_blocks = [results_block, cpu_time_block]
		step_contents_stats_apply = self._step_contents_element(
			content_position="sidebar_left", content_type="stats", 
			content_blocks=content_blocks)

		# build item
		item_id = self._build_item(grmlabcls=univariate, item_title=item_title,
			step_config=step_contents_config,
			step_stats_analysis=step_contents_stats_analysis,
			step_stats_apply=step_contents_stats_apply,
			summary_analysis_names=["table_num_ord", "table_cat_nom"])

		# build item for each variable in univariate
		json_variables = []
		udates = univariate._unique_dates
		for variable in univariate._variables_information:
			json_variables.append(self._add_univariate_variable(
				variable=variable, item_id=item_id, dates=udates))

		# add variables to json
		self._add_lines("data", json_variables)

	def add_optimalgrouping(self, optimalgrouping, 
		item_title="Optimal Grouping"):
		# check class type
		reporting_class_type_check("optimalgrouping", optimalgrouping, 
			OptimalGrouping)

		# optimalgrouping config options
		# ======================================================================
		config_data = {
			"index": [
				"special values handler policy",
				"missing values WoE policy",
				"special values WoE policy",
				"special values",
				"nominal variables",
				"parallel mode"
				],
			"data": [
				optimalgrouping._special_handler_policy,
				optimalgrouping._missing_woe_policy,
				optimalgrouping._special_woe_policy,
				"available" if optimalgrouping._special_values else "not available",
				"available" if optimalgrouping._variables_nominal else "not available",
				optimalgrouping._parallel
			]
		}

		config_block = self._content_blocks_element("config", config_data)

		# build config step contents
		content_blocks = [config_block]
		step_contents_config = self._step_contents_element(
			content_position="sidebar_left", content_type="config", 
			content_blocks=content_blocks)

		# feature binning analysis statistics
		# ======================================================================
		# db_info
		db_info_data = {
			"index": [
				"records",
				"columns",
				"target column",
				"target type"
				],
			"data": [
				optimalgrouping._n,
				optimalgrouping._nvars,
				optimalgrouping._target_name,
				optimalgrouping._target_type
				]
		}

		db_info_block = self._content_blocks_element("db_info_expanded", 
			db_info_data)

		# column analysis
		# ======================================================================
		column_analysis_data = {
			"index": [
				"numerical",
				"ordinal",
				"nominal",
				"categorical",
				"PD ascending",
				"PD descending",
				"PD peak",
				"PD valley",
				"PD undefined"
				],
			"data": [
				optimalgrouping._nvars_numerical,
				optimalgrouping._nvars_ordinal,
				optimalgrouping._nvars_nominal,
				optimalgrouping._nvars_categorical,
				optimalgrouping._nvars_ascending,
				optimalgrouping._nvars_descending,
				optimalgrouping._nvars_peak,
				optimalgrouping._nvars_valley,
				optimalgrouping._nvars_undefined
				]
		}

		column_analysis_block = self._content_blocks_element("column_analysis", 
			column_analysis_data)

		# cpu_time
		# ======================================================================
		if optimalgrouping._parallel:
			cpu_time_data = {
				"index": [
					"total",
					"processors"
					],
				"data": [
					optimalgrouping._time_analysis_total,
					optimalgrouping._cores
				]
			}
		else: 
			cpu_time_data = {
				"index": [
					"total",
					"numerical",
					"ordinal",
					"nominal",
					"categorical"
					],
				"data": [
					optimalgrouping._time_analysis_total,
					optimalgrouping._time_analysis_numerical,
					optimalgrouping._time_analysis_ordinal,
					optimalgrouping._time_analysis_nominal,
					optimalgrouping._time_analysis_categorical
					]
			}

		cpu_time_block = self._content_blocks_element("cpu_time", cpu_time_data)

		content_blocks = [db_info_block, column_analysis_block, cpu_time_block]
		step_contents_stats_analysis = self._step_contents_element(
			content_position="sidebar_left", content_type="stats", 
			content_blocks=content_blocks)
		
		# build item
		item_id = self._build_item(grmlabcls=optimalgrouping, 
			item_title=item_title, step_config=step_contents_config, 
			step_stats_analysis=step_contents_stats_analysis)

		# build item for each optbin in optimal grouping
		json_variables = []
		for variable in optimalgrouping._variables_information:
			json_variables.append(self._add_optbin(optbin=variable, 
				item_id=item_id))

		# add variable optbin to json
		self._add_lines("data", json_variables)

	def add_bivariate(self, bivariate, item_title="Bivariate Analysis"):

		# check class type
		reporting_class_type_check("bivariate", bivariate, Bivariate)

		# bivariate config options
		# ======================================================================
		config_data = {
			"index": [
				"special values handler policy",
				"missing values WoE policy",
				"special values WoE policy",
				"special values",
				"nominal values",
				"parallel mode",
				"optimal grouping provided"
				],
			"data": [
				bivariate._special_handler_policy,
				bivariate._special_woe_policy,
				bivariate._missing_woe_policy,
				"available" if bivariate._special_values else "not available",
				"available" if bivariate._variables_nominal else "not available",
				bivariate._parallel,
				"yes" if bivariate._is_optimalgrouping_provided else "no"
				]
		}

		config_block = self._content_blocks_element("config", config_data)

		# build config step contents
		content_blocks = [config_block]
		step_contents_config = self._step_contents_element(
			content_position="sidebar_left", content_type="config",
			content_blocks=content_blocks)

		# bivariate analysis statistics
		# ======================================================================
		db_info_data = {
			"index": [
				"records",
				"columns",
				"date column",
				"target column",
				"target type"
				],
			"data": [
				bivariate._n,
				bivariate._nvars,
				bivariate._date_name,
				bivariate._target_name,
				bivariate._target_type
				]
		}

		db_info_block = self._content_blocks_element("db_info_expanded",
			db_info_data)

		# column analysis
		# ======================================================================
		column_analysis_data = {
			"index": [
				"numerical",
				"ordinal",
				"nominal",
				"categorical",
				"low IV",
				"high IV",
				"low Gini",
				"high Gini",
				"hign p-value ANOVA test",
				"high p-value Chi2 test",
				"high max-pairwise p-value Chi2 test",
				"high p-value KS test",
				"high p-value Kruskal-Wallis test",
				"high p-value Median test",
				"low Cramer-V"
				],
			"data": [
				bivariate._nvars_numerical,
				bivariate._nvars_ordinal,
				bivariate._nvars_categorical,
				bivariate._nvars_nominal,
				bivariate._nvars_low_iv,
				bivariate._nvars_high_iv,
				bivariate._nvars_low_gini,
				bivariate._nvars_high_gini,
				bivariate._nvars_high_pvalue_anova,
				bivariate._nvars_high_pvalue_chi2,
				bivariate._nvars_high_pvalue_kolmogorov_smirnov,
				bivariate._nvars_high_pvalue_kruskal_wallis,
				bivariate._nvars_high_pvalue_median_t,
				bivariate._nvars_low_cramer_v
				]
		}

		column_analysis_block = self._content_blocks_element("column_analysis",
			column_analysis_data)

		# cpu_time
		# ======================================================================
		cpu_time_data0 = {
			"index": [
				"total",
				"optimal grouping",
				"bivariate metrics"
				],
			"data": [
				bivariate._time_analysis_total,
				bivariate._time_optimalgrouping,
				bivariate._time_bivariate_metrics
			]
		}

		if bivariate._parallel:
			cpu_time_data1 = {
				"index": [
					"total",
					"processors"
					],
				"data": [
					bivariate._time_analysis_total,
					bivariate._cores
					]
			}
		else:
			cpu_time_data1 = {
				"index": [
					"total",
					"numerical",
					"ordinal",
					"nominal",
					"categorical"
					],
				"data": [
					bivariate._time_bivariate_metrics,
					bivariate._time_analysis_numerical,
					bivariate._time_analysis_ordinal,
					bivariate._time_analysis_nominal,
					bivariate._time_analysis_categorical
					]
			}

		cpu_time_block0 = self._content_blocks_element("cpu_time", cpu_time_data0)
		cpu_time_block1 = self._content_blocks_element("cpu_time", cpu_time_data1)

		content_blocks = [db_info_block, column_analysis_block, cpu_time_block0,
			cpu_time_block1]
		step_contents_stats_analysis = self._step_contents_element(
			content_position="sidebar_left", content_type="stats",
			content_blocks=content_blocks)

		# bivariate apply statistics
		# ======================================================================
		# results
		results_data = {
			"index": [
				"original data",
				"after bivariate",
				"removed",
				"warnings"
				],
			"data": [
				bivariate._nvars,
				bivariate._nvars_after_apply,
				bivariate._nvars_removed,
				bivariate._nvars_warned
			]
		}

		results_block = self._content_blocks_element("results", results_data)

		# cpu time
		cpu_time_data = {
			"index": [
				"total",
				"remove"
				],
			"data": [
				bivariate._time_apply_total,
				bivariate._time_apply_remove
			]
		}

		cpu_time_block = self._content_blocks_element("cpu_time", cpu_time_data)

		content_blocks = [results_block, cpu_time_block]
		step_contents_stats_apply = self._step_contents_element(
			content_position="sidebar_left", content_type="stats",
			content_blocks=content_blocks)

		# build item
		item_id = self._build_item(grmlabcls=bivariate, item_title=item_title,
			step_config=step_contents_config,
			step_stats_analysis=step_contents_stats_analysis,
			step_stats_apply=step_contents_stats_apply)

		# build item for each variable in bivariate
		json_variables = []
		udates = bivariate._unique_dates
		for variable in bivariate._variables_information:
			optbin = bivariate._retrieve_optbin_variable(variable.name)
			json_variables.append(self._add_bivariate_variable(
				variable=variable, optbin=optbin, item_id=item_id,
				dates=udates))

		# add variables to json
		self._add_lines("data", json_variables)

	def add_model_generator(self, modelgenerator, item_title="Model Generator"):
		
		# check class type
		reporting_class_type_check("modelgenerator", modelgenerator,
			ModelGenerator)

		# modelgenerator config options
		# ======================================================================
		config_data = {
			"index": [
				"test size",
				"random state",
				"noise factor"
				],
			"data": [
				modelgenerator._split_test_size,
				modelgenerator._split_seed,
				modelgenerator._noise_factor
				]
		}

		config_block = self._content_blocks_element("config", config_data)

		# build config step contents
		content_blocks = [config_block]
		step_contents_config = self._step_contents_element(
			content_position="sidebar_left", content_type="config",
			content_blocks=content_blocks)

		# modelgenerator analysis statistics
		# ======================================================================
		db_info_data = {
			"index": [
				"records",
				"columns",
				"target column",
				"target type",
				],
			"data": [
				modelgenerator._n,
				modelgenerator._nvars,
				modelgenerator._target_name,
				modelgenerator._target_type
			]
		}

		db_info_block = self._content_blocks_element("db_info_expanded", 
			db_info_data)

		# models analysis
		# ======================================================================
		models_analysis_data = {
			"index": [
				"records train set",
				"records test set",
				"models",
				"feature selection",
				"estimators"
				],
			"data": [
				modelgenerator._n_train,
				modelgenerator._n_test,
				modelgenerator._n_models,
				modelgenerator._n_feature_selection,
				modelgenerator._n_estimators
				]
		}

		models_analysis_block = self._content_blocks_element("details",
			models_analysis_data)

		# cpu time
		# ======================================================================
		rest = modelgenerator._time_total_feature_selection
		rest += modelgenerator._time_total_estimators
		rest += modelgenerator._time_total_metrics
		others = modelgenerator._time_total - rest
		cpu_time_data = {
			"index": [
				"total",
				"feature selection",
				"estimators",
				"model metrics",
				"others"
				],
			"data": [
				modelgenerator._time_total,
				modelgenerator._time_total_feature_selection,
				modelgenerator._time_total_estimators,
				modelgenerator._time_total_metrics,
				others
				]
		}

		cpu_time_block = self._content_blocks_element("cpu_time", cpu_time_data)

		content_blocks = [db_info_block, models_analysis_block, cpu_time_block]
		step_contents_stats_analysis = self._step_contents_element(
			content_position="sidebar_left", content_type="stats", 
			content_blocks=content_blocks)

		# build item
		self._build_item(grmlabcls=modelgenerator, 
			item_title=item_title, step_config=step_contents_config, 
			step_stats_analysis=step_contents_stats_analysis)

	def _add_univariate_variable(self, variable, item_id, dates):
		# variable analysis corresponding to analysis step ("01")
		step_id = "01"
		content_position = 2

		univariate_common = {
			"name": variable.name,
			"type": variable.dtype,
			"dates": dates,
			"informed records": variable.n_informed,
			"missing records": variable.n_missing,
			"special records": variable.n_special,
			"unique special values": variable.unique_special,
			"months w/o information": variable.t_n_months_no_info,
			"information stability (weighted)": variable.t_stab_info_desc,
			"information stability (evenly)": variable.t_stab_info_prop,
			"data values stability (weighted)": variable.t_stab_desc,
			"data values stability (evenly)": variable.t_stab_prop,
			"hist_info_col": variable.d_hist_columns,
			"hist_info_pos": variable.d_hist_positions,
			"temp_missing": variable.t_missing,
			"temp_special": variable.t_special,
			"temp_info": variable.t_info
		}

		if variable.dtype in ("numerical", "ordinal"):
			block_position = 0
			univariate_dtype = {
				"zeros": variable.d_nzero,
				"zeros (%)": variable.d_perc_zero,
				"negative": variable.d_nneg,
				"positive": variable.d_npos,
				"min": variable.d_min,
				"max": variable.d_max,
				"mean": variable.d_mean,
				"std": variable.d_std,
				"mode": variable.d_mode,
				"percentile 1%": variable.d_percentile1,
				"percentile 25%": variable.d_percentile25,
				"median": variable.d_median,
				"percentile 75%": variable.d_percentile75,
				"percentile 99%": variable.d_percentile99,
				"IQR": variable.d_iqr,
				"outliers low": variable.d_n_outlier_low,
				"outliers high": variable.d_n_outlier_high,
				"outlier low threshold": variable.d_outlier_low_threshold,
				"outlier high threshold": variable.d_outlier_high_threshold,
				"concentration": variable.d_concentration,
				"concentration interval": variable.d_concentration_interval,
				"dispersion_parametric": variable.d_coeff_dispersion_parametric,
				"dispersion_nonparametric": variable.d_coeff_dispersion_nonparametric,
				"temp_mean": variable.t_mean,
				"temp_percentile25": variable.t_percentile25,
				"temp_median": variable.t_median,
				"temp_percentile75": variable.t_percentile75,
			}
		else:
			block_position = 1
			univariate_dtype = {
				"categories": variable.d_n_categories,
				"most frequent category": variable.d_most_freq_cat,
				"most frequent category (%)": variable.d_perc_most_freq_cat,
				"HHI": variable.d_concentration_hhi,
				"HHI (normalized)": variable.d_concentration_hhi_norm,
				"top_cat": variable.d_top_cat,
				"top_cat_n": variable.d_top_n_cat,
				"temp_c0": variable.t_c0,
				"temp_c1": variable.t_c1,
				"temp_c2": variable.t_c2,
				"temp_c3": variable.t_c3,
				"temp_c4": variable.t_c4,
				"temp_c5": variable.t_c5,
				"temp_c6": variable.t_c6,
				"temp_c7": variable.t_c7,
				"temp_c8": variable.t_c8,
				"temp_c9": variable.t_c9,
				"temp_rest": variable.t_rest
			}

		# merge dictionaries
		univariate_variable = {**univariate_common, **univariate_dtype}

		# variable structure
		json_univariate = {
			"item_id": item_id,
			"step_id": step_id,
			"content_position": content_position,
			"block_position": block_position,
			"data": univariate_variable
		}

		variable_id = "{}_{}_{}_{}_{}".format(item_id, step_id, content_position, 
			block_position, variable.name)

		str_json_variable = _JSON_DATA_ITEMS.format(variable_id, 
			dict_to_json(json_univariate))
		
		return str_json_variable

	def _add_optbin(self, optbin, item_id):
		# variable optimal binning. Analysis step ("01")
		step_id = "01"
		content_position = 2
		block_position = 0

		# auxiliar parameters
		prebinning = "yes" if optbin._is_prebinning else "no"
		minbs = "not set" if optbin._min_buckets is None else optbin._min_buckets
		maxbs = "not set" if optbin._max_buckets is None else optbin._max_buckets 

		if optbin._monotonicity_sense is "ascending":
			sense = "asc"
		elif optbin._monotonicity_sense is "descending":
			sense = "desc"
		elif optbin._monotonicity_sense is "concave":
			sense = "concave"
		elif optbin._monotonicity_sense is "convex":
			sense = "convex"
		elif optbin._monotonicity_sense is "peak":
			sense = "peak"
		elif optbin._monotonicity_sense is "valley":
			sense = "valley"
		elif optbin._monotonicity_sense is "undefined":
			# TODO: check undefined / auto message
			sense = "undefined"

		if optbin._monotonicity_user is "ascending":
			user_sense = "asc"
		elif optbin._monotonicity_user is "descending":
			user_sense = "desc"
		elif optbin._monotonicity_user is "concave":
			user_sense = "concave"
		elif optbin._monotonicity_user is "convex":
			user_sense = "convex"
		elif optbin._monotonicity_user is "peak":
			user_sense = "peak"
		elif optbin._monotonicity_user is "valley":
			user_sense = "valley"			
		else:
			user_sense = "auto"

		spec = "yes" if optbin._special_values else "no"
		prebuckets = "yes" if optbin._user_splits_provided else "no"
		indexforced = "yes" if optbin._user_idx_forced_splits else "no"
		regularization = "yes" if optbin._regularization else "no"
		reduce_bucket_diff = "yes" if optbin._reduce_bucket_size_diff else "no"

		# pre-binning algorithm options
		if optbin._prebinning_algorithm is "ctree":
			ct_vartype = optbin._ctree_variable_type
			ct_mincrit = round(optbin._ctree_min_criterion, 8)
			ct_maxcand = optbin._ctree_max_candidates
			ct_dynamic = optbin._ctree_dynamic_split_method
		else:
			ct_vartype = "not set"
			ct_mincrit = "not set"
			ct_maxcand = "not set"
			ct_dynamic = "not set"

		if not optbin._is_prebinning:
			pre_algo = "not set"
		else:
			pre_algo = optbin._prebinning_algorithm
		
		# optimization problem status
		status = "stopped"
		if optbin._is_solution_optimal:
			status = "optimal"
		elif optbin._is_infeasible:
			status = "infeasible"
		elif optbin._is_unbounded:
			status = "unbounded"

		# cuts are not always generated
		if not optbin._cuts_generated:
			cut_g = 0
			cut_u = 0
		else:
			cut_g = optbin._cuts_generated
			cut_u = optbin._cuts_used

		# buckets reduction
		buckets_reduction = optbin._n_optimal_buckets - optbin._n_prebuckets

		if optbin._n_prebuckets >= 2:
			infeas_ratio = optbin._infeasible_buckets / optbin._n_prebuckets
		else:
			infeas_ratio = 0

		# optbin dictionary structure
		optbin_variable = {
			"name": optbin._name,
			"type": optbin._dtype,
			"prebinning": optbin._is_prebinning,
			"prebinning_max_nodes": optbin._prebinning_max_leaf_nodes,
			"monotonicity_sense": user_sense,
			"min_buckets": minbs,
			"max_buckets": maxbs,
			"min_bucket_size": optbin._min_bucket_size,
			"max_bucket_size": optbin._max_bucket_size,
			"min_pd_difference": optbin._min_pd_diff,
			"regularization": regularization,
			"reduce_bucket_size_diff": reduce_bucket_diff,
			"special_values": spec,
			"special_handler_policty": optbin._special_handler_policy,
			"special_woe_policy": optbin._special_woe_policy,
			"missing_woe_policy": optbin._missing_woe_policy,
			"pre-buckets": prebuckets,
			"indexes_forced": indexforced,
			"algorithm": pre_algo,
			"variable_type": ct_vartype,
			"min_criterion": ct_mincrit,
			"max_candidates": ct_maxcand,
			"dynamic_split_method": ct_dynamic,
			"original_variables": optbin._nvs,
			"original_constraints": optbin._ncs,
			"original_nonzeros": optbin._nnz,
			"after_variables": optbin._nvs_preprocessing,
			"after_constraints": optbin._ncs_preprocessing,
			"after_nonzeros": optbin._nnz_preprocessing,
			"status": status,
			"objective": optbin._mipcl_obj,
			"infeasible_buckets": optbin._infeasible_buckets,
			"infeasible_buckets_perc": infeas_ratio,
			"cuts_generated": cut_g,
			"cuts_used": cut_u,
			"branch_and_cut_nodes": optbin._branch_and_cut_nodes,
			"time_total": optbin._time_total,
			"time_prebinning": optbin._time_prebinning,
			"time_model_data": optbin._time_problem_data,
			"time_model_generation": optbin._time_problem_generation,
			"time_optimizer": optbin._time_optimizer,
			"time_preprocessing": optbin._time_optimizer_preprocessing,
			"post_analysis": optbin._time_post_analysis,
			"root_LP_algorithm": "Dual-Simplex",
			"time_limit": optbin._time_limit,
			"MIP_gap": 0,
			"prebinning_buckets": optbin._n_prebuckets,
			"prebinning_iv": optbin._iv_prebinning,
			"prebinning_trend_changes": optbin._prebinning_trend_changes,
			"optimal_buckets": optbin._n_optimal_buckets,
			"optimal_iv": optbin._iv_optimal,
			"optimal_monotonicity_sense": sense,
			"pvalue": optbin._max_pvalue,
			"pvalue_method": optbin._pvalue_method,
			"smallest_bucket_perc": optbin._smallest_bucket_perc,
			"smallest_bucket": optbin._smallest_bucket_size,
			"largest_bucket_perc": optbin._largest_bucket_perc,
			"largest_bucket": optbin._largest_bucket_size,
			"std_bucket_size": optbin._std_bucket_size,
			"group_missing": optbin._group_missing,
			"group_special": optbin._group_special,
			"group_others": optbin._group_others,
			"binning_table": json.loads(optbin.optbin_result().to_json(
				orient="split", index=False))
		}

		# optbin structure
		json_optbin = {
			"item_id": item_id,
			"step_id": step_id,
			"content_position": content_position,
			"block_position": block_position,
			"data": optbin_variable
		}

		optbin_id = "{}_{}_{}_{}_{}".format(item_id, step_id, content_position,
			block_position, optbin._name)

		str_json_optbin = _JSON_DATA_ITEMS.format(optbin_id, dict_to_json(
			json_optbin))

		return str_json_optbin

	def _add_bivariate_variable(self, variable, optbin, item_id, dates):
		# variable bivariate analysis. Analysis step ("01")
		step_id = "01"
		content_position = 2
		block_position = 0

		# binning tables: pandas dataframe ==> json
		json_basic_binning_table = json.loads(
			variable.basic_binning_table.to_json(orient="split", index=False))
		json_optimal_binning_table = json.loads(optbin.optbin_result().to_json(
			orient="split", index=False))

		# bivariate variable common attributes
		bivariate_common = {
			"name": variable.name,
			"type": variable.dtype,
			"dates": dates,
			"iv": variable.iv,
			"gini": variable.gini,
			"pd_monotonicity": optbin._monotonicity_sense,
			"groups": optbin._n_optimal_buckets,
			"group_special": optbin._group_special,
			"group_missing": optbin._group_missing,
			"group_others": optbin._group_others,
			"pvalue_chi2": variable.pvalue_chi2,
			"pvalue_chi2_max": variable.pvalue_chi2_max,
			"cramer_v": variable.cramer_v,
			"basic_binning_table": json_basic_binning_table,
			"optimal_binning_table": json_optimal_binning_table,
			"temp_grp_count_by_date": variable.t_grp_count_by_date,
			"temp_grp_event_by_date": variable.t_grp_event_by_date,
			"temp_n_grp_empty": variable.t_n_grp_empty,
			"temp_n_months_no_event": variable.t_n_months_no_event,
			"temp_pd_mean": variable.t_pd_mean,
			"temp_pd_std": variable.t_pd_std,
			"temp_pd_min": variable.t_pd_min,
			"temp_pd_p25": variable.t_pd_p25,
			"temp_pd_median": variable.t_pd_median,
			"temp_pd_p75": variable.t_pd_p75,
			"temp_pd_max": variable.t_pd_max,
			"temp_pd_dispersion": variable.t_pd_dispersion,
			"temp_iv_mean": variable.t_iv_mean,
			"temp_iv_std": variable.t_iv_std,
			"temp_iv_min": variable.t_iv_min,
			"temp_iv_p25": variable.t_iv_p25,
			"temp_iv_median": variable.t_iv_median,
			"temp_iv_p75": variable.t_iv_p75,
			"temp_iv_max": variable.t_iv_max,
			"temp_iv_dispersion": variable.t_iv_dispersion,
			"temp_grp_size_stab_global": variable.t_grp_size_stability_global,
			"temp_grp_size_stab_weighted": variable.t_grp_size_stability_weighted
		}

		if variable.dtype in ("numerical", "ordinal"):
			bivariate_dtype = {
			"pvalue_anova": variable.pvalue_anova,
			"pvalue_kolmogorov_smirnov": variable.pvalue_kolmogorov_smirnov,
			"pvalue_kruskal_wallis": variable.pvalue_kruskal_wallis,
			"pvalue_median_t": variable.pvalue_median_t,
			"divergence": variable.divergence,
			"wasserstein_distance": variable.wasserstein_distance,
			"mean_nonevent": variable.mean_nonevent,
			"std_nonevent": variable.std_nonevent,
			"min_nonevent": variable.min_nonevent,
			"p25_nonevent": variable.percentile_25_nonevent,
			"median": variable.median_nonevent,
			"p75_nonevent": variable.percentile_75_nonevent,
			"max_nonevent": variable.max_nonevent,
			"mean_event": variable.mean_event,
			"std_event": variable.std_event,
			"min_event": variable.min_event,
			"p25_event": variable.percentile_25_event,
			"median_event": variable.median_event,
			"p75_event": variable.percentile_75_event,
			"max_event": variable.max_event
			}
		else:
			bivariate_dtype = {
			"categories": variable.n_categories,
			"most frequent category": variable.most_freq_cat,
			"most frequent category (%)": variable.perc_most_freq_cat,
			"mean_number_categories": variable.mean_number_cat,
			"std_number_categories": variable.std_number_cat,
			"min_number_categories": variable.min_number_cat,
			"p25_number_categories": variable.p25_number_cat,
			"median_number_categories": variable.median_number_cat,
			"p75_number_categories": variable.p75_number_cat,
			"max_number_categories": variable.max_number_cat
			}

		# merge dictionaries
		bivariate_variable = {**bivariate_common, **bivariate_dtype}

		# bivariate structure
		json_bivariate = {
			"item_id": item_id,
			"step_id": step_id,
			"content_position": content_position,
			"block_position": block_position,
			"data": bivariate_variable
		}

		bivariate_id = "{}_{}_{}_{}_{}".format(item_id, step_id,
			content_position, block_position, variable.name)

		str_json_bivariate = _JSON_DATA_ITEMS.format(bivariate_id,
			dict_to_json(json_bivariate))

		return str_json_bivariate

	def _summary_tables(self, tables, table_names):
		"""
		Handle cases with multiple summary tables, e.g. univariate analysis.
		"""
		if table_names is not None:
			if len(tables) != len(table_names):
				raise ValueError("tables and table_names different size.")
		else:
			table_names = ["table_{}".format(i) for i in range(len(tables))]

		summary_tables = []
		for i, table in enumerate(tables):
			table_name = table_names[i]
			sum_table = self._content_blocks_element(table_name, table, True)
			summary_tables.append(sum_table)

		return summary_tables

	def _build_item(self, grmlabcls, item_title, step_config=None,
		step_stats_analysis=None, step_stats_apply=None, 
		summary_analysis_names=None, summary_apply_names=None, 
		extra_item_step=None):
		"""Build item information."""
		if grmlabcls._is_analysis_run:
			item_steps = []
			step_analysis_contents = []

			# retrieve information from GRMlab class method config()
			if step_config is not None:
				contents_config = step_config
				step_analysis_contents.append(contents_config)

			# retrieve information from GRMlab class method analysis()
			if step_stats_analysis is not None:
				contents_stats = step_stats_analysis
				step_analysis_contents.append(contents_stats)

			# retrieve information from GRMlab class method summary_analysis()
			# convert pandas json to content block structure
			summary_data = grmlabcls.summary_analysis("json")

			if isinstance(summary_data, (tuple, list)):
				summary_tables = self._summary_tables(summary_data, 
					summary_analysis_names)
			else:
				summary = self._content_blocks_element("table", summary_data, True)
				summary_tables = [summary]

			contents_summary = self._step_contents_element(
				content_position="mainbody", content_type="summary", 
				content_blocks=summary_tables)
			step_analysis_contents.append(contents_summary)

			# build item step
			item_analysis = self._item_steps_element(step_id="01", 
				step_type="analysis", step_contents=step_analysis_contents)
			item_steps.append(item_analysis)

		if hasattr(grmlabcls, '_is_apply_run') and grmlabcls._is_apply_run:
			step_apply_contents = []

			# retrieve information form class method apply
			if step_stats_apply is not None:
				contents_stats = step_stats_apply
				step_apply_contents.append(contents_stats)

			# retrieve information from GRMlab class method summary_apply()
			# convert pandas json to content block structure
			summary_data = grmlabcls.summary_apply("json")

			if isinstance(summary_data, (tuple, list)):
				summary_tables = self._summary_tables(summary_data, 
					summary_apply_names)
			else:
				summary = self._content_blocks_element("table", summary_data, True)
				summary_tables = [summary]
			
			contents_summary = self._step_contents_element(
				content_position="mainbody", content_type="summary", 
				content_blocks=summary_tables)
			step_apply_contents.append(contents_summary)

			# build item step
			item_apply = self._item_steps_element(step_id="02", 
				step_type="apply", step_contents=step_apply_contents)
			item_steps.append(item_apply)

		# build item element
		item_type = grmlabcls.__class__.__name__.lower()
		item_id, json_item = self._item_element(item_type=item_type, 
			item_layout_version="01", item_steps=item_steps)

		# add items to data.json and contents.json
		self._add_item_data(json_item)
		self._add_item_contents(json_item, item_title)

		# by default return item_id => needed for individual json variable
		return item_id

	def _init_report(self):
		"""Initialize GRMlab report."""
		
		# path format according to o.s.
		abspath = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
		system_os = platform.system()
		linux_os = (system_os == "Linux" or "CYGWIN" in system_os)

		if system_os == "Windows":
			# NOTE: ananconda accepts linux style paths, but we cannot access
			# files without proper windows paths.
			dest = self._path.replace("/", "\\")
		else:
			dest = self._path
			
		# check if folder structure already corresponds to grmlab report
		path_is_report = is_report(dest, system_os)
		
		if not path_is_report:
			# new report
			if linux_os:
				source = abspath+"/reporting/report/"
			else:
				source = abspath+"\\reporting\\report\\"

			# generate list of files to be copied
			shutil.copytree(source, dest, copy_function=shutil.copy)

		# save paths to contents.json and data.json
		if linux_os:
			self._json_data_file = dest+"/medatadata/data.json"
			self._json_contents_file = dest+"/metadata/contents.json"
		else:
			self._json_data_file = dest+"\\metadata\\data.json"
			self._json_contents_file = dest+"\\metadata\\contents.json"

		# initialize json files
		if not path_is_report:
			self._init_json_data()

	def _report_info(self):
		"""Return report information summary."""
		report_info = {
			"title1": self._report_title,
			"title2": self._report_description,
			"title3": self._report_date
		}

		return report_info

	def _init_json_data(self):
		"""Initialize JSON files: data.json and contents.json."""
		self._add_line("data", _JSON_DATA_ITEMS_OBJECT)
		self._add_line("contents", _JSON_CONTENTS_REPORT_INFO.format(
			dict_to_json(self._report_info())))
		self._add_line("contents", _JSON_CONTENTS_USER_INFO.format(
			dict_to_json(user_info())))
		self._add_line("contents", _JSON_CONTENTS_ITEMS_ARRAY)

	def _add_line(self, file, line):
		"""Add (append) line to json file."""
		if file == "data":
			file = self._json_data_file
		elif file == "contents":
			file = self._json_contents_file

		try:
			with open(file, 'a') as f:
				f.write(line+"\n")
		except EnvironmentError as err:
			print("Error: {}".format(err))
			raise
		except:
			print("Unexpected error: {}".format(sys.exc_info()[0]))
			raise

	def _add_lines(self, file, lines):
		"""Add several lines at once to json file."""
		if file == "data":
			file = self._json_data_file
		elif file == "contents":
			file = self._json_contents_file

		try:
			with open(file, 'a') as f:
				for line in lines:
					f.write(line+"\n")
		except EnvironmentError as err:
			print("Error: {}".format(err))
			raise
		except:
			print("Unexpected error: {}".format(sys.exc_info()[0]))
			raise

	def _add_item_data(self, json_item):
		"""Add item to file data.json."""
		id = json_item["item_id"]
		str_json_data = _JSON_DATA_ITEMS.format(id, dict_to_json(json_item))
		self._add_line("data", str_json_data)

	def _add_item_contents(self, json_item, item_title):
		"""Add item to file contents.json."""
		item_id = json_item["item_id"]
		
		# main item
		json_content = {
			"item_id": item_id,
			"item_info": 0,
			"step_id": "00",
			"title": item_title,
			"type": json_item["item_type"]
		}

		# subitems (analysis/apply etc...)
		str_json_content = _JSON_CONTENTS_ITEMS.format(
			dict_to_json(json_content))
		self._add_line("contents", str_json_content)

		# each item step adds a new line in contents.json
		for item_step in json_item["item_steps"]:
			json_content = {
				"item_id": item_id,
				"step_id": item_step["step_id"],
				"title": item_step["step_type"].capitalize(), 
				"type": item_step["step_type"]
			}
		
			str_json_content = _JSON_CONTENTS_ITEMS.format(
				dict_to_json(json_content))
			self._add_line("contents", str_json_content)

	def _item_element(self, item_type, item_layout_version, item_steps):
		"""Return item element json and corresponding item_id."""
		item_id = get_id(self._json_contents_file)
		json_item = {
			"item_id": item_id,
			"item_type": item_type,
			"item_layout_version": item_layout_version,
			"item_steps": item_steps
		}
		
		return item_id, json_item

	@staticmethod
	def _item_steps_element(step_id, step_type, step_contents):
		json_item_steps = {
			"step_id": step_id,
			"step_type": step_type,
			"step_contents": step_contents
		}

		return json_item_steps

	def _step_contents_element(self, content_position, content_type, 
		content_blocks):
		"""Generate step content element as a dictionary."""
		json_step_contents = {
			"content_position": content_position,
			"content_type": content_type,
			"content_blocks": content_blocks
		}

		return json_step_contents

	@staticmethod
	def _content_blocks_element(block_type, block_data, is_json=False):
		"""
		Generate content block element as a dictionary. If block_data is a 
		pandas json output => deserialize.
		"""
		bk_data = block_data
		if is_json:
			bk_data = json.loads(block_data)

		json_content_blocks = {
			"block_type": block_type,
			"block_data": bk_data
		}

		return json_content_blocks
