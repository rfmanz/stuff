"""
Bivariate analysis for continuous target.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import gc
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import kendalltau
from scipy.stats import mannwhitneyu
from scipy.stats import median_test
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import ttest_ind_from_stats
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

from ..core.dtypes import check_date_format
from ..core.dtypes import check_dtype
from ..core.dtypes import check_target_dtype
from ..core.dtypes import is_binary
from ..core.dtypes import is_numpy_float
from ..core.dtypes import is_numpy_int
from ..core.dtypes import is_numpy_object
from ..core.exceptions import NotRunException
from ..data_processing.feature_binning import OptimalGrouping
from ..data_processing.feature_binning import plot
from ..data_processing.feature_binning import table
from ..data_processing.feature_binning.util import process_data
from ..reporting.util import reporting_output_format
from .bivariate import Bivariate
from .util import js_divergence_multivariate


STATUS_PVALUE_WELCH_T_HIGH = "pvalue_welch_t_high"
STATUS_PVALUE_MANN_WHITNEY_HIGH = "pvalue_mann_whitney_high"
STATUS_PVALUE_MEDIAN_HIGH = "pvalue_median_high"
STATUS_CORR_PEARSON_LOW = "corr_pearson_low"
STATUS_CORR_KENDALLTAU_LOW = "corr_kendalltau_low"
STATUS_CORR_SPEARMAN_LOW = "corr_spearman_low"
STATUS_R2_SCORE_LOW = "r2_score_low"
STATUS_UNDEFINED = "undefined"

STATUS_OK = "ok"

STATUS_OPTIONS = [STATUS_PVALUE_WELCH_T_HIGH, STATUS_PVALUE_MANN_WHITNEY_HIGH,
	STATUS_PVALUE_MEDIAN_HIGH, STATUS_CORR_PEARSON_LOW, STATUS_OK,
	STATUS_CORR_KENDALLTAU_LOW, STATUS_CORR_SPEARMAN_LOW, STATUS_UNDEFINED,
	STATUS_R2_SCORE_LOW]

STATUS_REMOVE = [STATUS_PVALUE_WELCH_T_HIGH, STATUS_PVALUE_MANN_WHITNEY_HIGH,
	STATUS_PVALUE_MEDIAN_HIGH, STATUS_R2_SCORE_LOW, STATUS_UNDEFINED]

STATUS_REVIEW = [STATUS_CORR_PEARSON_LOW, STATUS_CORR_KENDALLTAU_LOW,
	STATUS_CORR_SPEARMAN_LOW]


def _compute_pvalues(x, y, splits, special_values, tb):
	# parametric test
	if len(tb) <= 1:
		max_pvalue_welch_t = 0
		max_pvalue_mann_whitney = 0
		max_pvalue_median = 0
	else:
		max_pvalue_welch_t = max(ttest_ind_from_stats(tb[i, 1], tb[i, 2],
			tb[i, 0], tb[i+1, 1], tb[i+1, 2], tb[i+1, 0], equal_var=False)[1]
			for i in range(len(tb)-1))

		# non-parametric tests
		x, y, _ = process_data(x, y, None, special_values)

		max_pvalue_median = 0
		max_pvalue_mann_whitney = 0

		if isinstance(splits[0], (list, np.ndarray)):
			for i in range(len(splits) - 1):
				y1 = y[pd.Series(x).isin(splits[i]).values]
				y2 = y[pd.Series(x).isin(splits[i+1]).values]

				pvalue_mann_whitney = mannwhitneyu(y1, y2)[1]
				pvalue_median = median_test(y1, y2)[1]

				if pvalue_mann_whitney > max_pvalue_mann_whitney:
					max_pvalue_mann_whitney = pvalue_mann_whitney

				if pvalue_median > max_pvalue_median:
					max_pvalue_median = pvalue_median
		else:
			bands = np.insert(splits, 0, np.min(x))
			bands = np.append(bands, np.max(x))

			for i in range(len(bands) - 2):
				if i == 0:
					y1 = y[(x <= bands[i+1])]
				else:
					y1 = y[(x > bands[i]) & (x <= bands[i+1])]

				y2 = y[(x > bands[i+1]) & (x <= bands[i+2])]

				pvalue_mann_whitney = mannwhitneyu(y1, y2)[1]
				pvalue_median = median_test(y1, y2)[1]

				if pvalue_mann_whitney > max_pvalue_mann_whitney:
					max_pvalue_mann_whitney = pvalue_mann_whitney

				if pvalue_median > max_pvalue_median:
					max_pvalue_median = pvalue_median

	return max_pvalue_welch_t, max_pvalue_mann_whitney, max_pvalue_median


def _variable_descriptive_numerical(name, data):
	report = (
	"---------------------------------------------------------------------\n"
	"                         Descriptive analysis                        \n"
	"---------------------------------------------------------------------\n"
	"   variable: {}\n"
	"\n"
	"   Binning information              Statistics\n"
	"     R^2                {:>7.4f}       p-value Welch-t         {:>6.5f}\n"
	"     MAE              {:>.3E}       p-value Mann-Whitney    {:>6.5f}\n"
	"     monotonicity    {:>10}       p-value median          {:>6.5f}\n"
	"     groups          {:>10}       correlation Pearson     {:>6.4f}\n"
	"     group special   {:>10}       correlation Spearman    {:>6.4f}\n"
	"     group missing   {:>10}       correlation Kendall-tau {:>6.4f}\n"
	"     group others    {:>10}\n"
	"---------------------------------------------------------------------\n"
	)

	keys = ["r2_score", "pvalue_welch_t", "mean_absolute_error",
		"pvalue_mann_whitney", "monotonicity", "pvalue_median", "groups",
		"corr_pearson", "group_special", "corr_spearman", "group_missing",
		"corr_kendalltau", "group_others"]

	report_data = [name] + list(data[k] for k in keys)

	return report.format(*report_data)


def _variable_descriptive_categorical(name, data):
	report = (
	"---------------------------------------------------------------------\n"
	"                         Descriptive analysis                        \n"
	"---------------------------------------------------------------------\n"
	"   variable: {}\n"
	"\n"
	"   Binning information              Statistics\n"
	"     R^2                {:>7.4f}       p-value Welch-t         {:>6.5f}\n"
	"     MAE              {:>.3E}       p-value Mann-Whitney    {:>6.5f}\n"
	"     monotonicity    {:>10}       p-value median          {:>6.5f}\n"
	"     groups          {:>10}       correlation Pearson     {:>6.4f}\n"
	"     group special   {:>10}       correlation Spearman    {:>6.4f}\n"
	"     group missing   {:>10}       correlation Kendall-tau {:>6.4f}\n"
	"     group others    {:>10}       number of categories     {:>6}\n"
	"                                      most freq. category  {:>10}\n"
	"                                      % most freq. category   {:>7.2%}\n"
	"\n"
	"   Distribution (number of categories by group)\n"
	"     mean                {:>6.1f}\n"
	"     std                 {:>6.1f}\n"
	"     min                 {:>6}\n"
	"     p25                 {:>6.1f}\n"
	"     p50                 {:>6.1f}\n"
	"     p75                 {:>6.1f}\n"
	"     max                 {:>6}\n"
	"---------------------------------------------------------------------\n"
	)

	keys = ["r2_score", "pvalue_welch_t", "mean_absolute_error",
		"pvalue_mann_whitney", "monotonicity", "pvalue_median", "groups",
		"corr_pearson", "group_special", "corr_spearman", "group_missing",
		"corr_kendalltau", "group_others", "n_categories", "most_freq_category",
		"p_most_freq_category", "mean_n_categories", "std_n_categories",
		"min_n_categories", "percentile_25_n_categories", "median_n_categories",
		"percentile_75_n_categories", "max_n_categories"]

	report_data = [name] + list(data[k] for k in keys)

	return report.format(*report_data)


def _variable_temporal(name, data):
	report = (
	"---------------------------------------------------------------------\n"
	"                          Temporal analysis                          \n"
	"---------------------------------------------------------------------\n"
	"   variable: {}\n"
	"\n"
	"   group size divergence\n"
	"     uniform        {:>5.3f}               exponential      {:>5.3f}\n"
	"\n"
	"   Distribution      (R^2 by split)\n"
	"     mean                {:>6.3f}\n"
	"     std                 {:>6.3f}\n"
	"     min                 {:>6.3f}\n"
	"     p25                 {:>6.3f}\n"
	"     p50                 {:>6.3f}\n"
	"     p75                 {:>6.3f}\n"
	"     max                 {:>6.3f}\n"
	"---------------------------------------------------------------------\n"
	)

	keys = ["t_split_div_uniform", "t_split_div_exponential", "t_r2_mean",
		"t_r2_std", "t_r2_min", "t_r2_percentile_25", "t_r2_median",
		"t_r2_percentile_75", "t_r2_max"]

	report_data = [name] + list(data[k] for k in keys)
	return report.format(*report_data)	


def _variable_plot_descriptive(optbin, basic_binning_table, basic_splits,
	metric):

	if metric is not None:
		if metric not in ("mean", "median"):
			raise ValueError("metric {} not supported.".format(metric))

		lb_feature = "Mean" if metric == "mean" else "Median"
	else:
		lb_feature = "Mean" if optbin.metric == "mean" else "Median"

	# plot optimal binning
	optbin.plot_binning_table(metric)

	# plot fixed binning variables
	plot(basic_binning_table, basic_splits, plot_type="default",
		lb_feature=lb_feature)


def _variable_plot_temporal(name, data, metric):
	if metric == "mean":
		split_metric = data["t_split_mean_by_date"]
	else:
		split_metric = data["t_split_median_by_date"]

	split_size = data["t_split_count_by_date"] / data["t_split_count_by_date"].sum(axis=0)
	n_splits, _ = split_metric.shape

	plt.subplot(2, 1, 1)
	for i in range(n_splits):
		plt.plot(split_metric[i, :], label="split {}".format(i))
	plt.ylabel(metric.capitalize())

	plt.subplot(2, 1, 2)
	for i in range(n_splits):
		plt.plot(split_size[i, :], label="split {}".format(i))
	plt.ylabel("Group size")
	plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), 
		fancybox=True, shadow=True, ncol=3)

	plt.show()

	plt.boxplot(split_metric.tolist(), labels=["split {}".format(i) for i in 
		range(n_splits)])
	plt.ylabel(metric.capitalize())

	plt.show()


def _bivariate_continuous_results(dict_variables, step, date, format):
	"""Return run or transform step results as a dataframe or json."""

	# common metrics
	order_cols = ["name", "dtype", "recommended_action", "status", "comment",
		"mean_absolute_error", "corr_spearman", "r2_score", "monotonicity",
		"groups", "group_special", "group_missing", "group_others",
		"pvalue_welch_t", "pvalue_mann_whitney", "pvalue_median", "corr_pearson",
		"corr_kendalltau", "optbin_params", "optbin_comment", "optbin_default"]

	order_cols_transform = ["name", "dtype", "action", "status", "comment",
		"recommended_action", "user_action", "user_comment", "auto_comment"]

	results = pd.DataFrame.from_dict(dict_variables).T
	results.reset_index(level=0, inplace=True)
	results.rename(columns={"index": "name"}, inplace=True)

	if step == "run":
		special_cols = ["comment", "optbin_params", "optbin_comment",
			"optbin_default"]

		for col in special_cols:
			if not col in results.columns:
				results[col] = np.nan

		results = results[order_cols]
		return reporting_output_format(results, format)
	else:
		for col in ["comment", "user_action", "user_comment", "auto_comment"]:
			if not col in results.columns:
				results[col] = np.nan

		results = results[order_cols_transform]
		return reporting_output_format(results, format)


def _bivariate_continuous_stats(report_data, step):
	if step == "run":
		report = (
		"\033[94m================================================================================\033[0m\n"
		"\033[1m\033[94m                       GRMlab Bivariate Continuous 0.1: Run                            \033[0m\n"
		"\033[94m================================================================================\033[0m\n"
		"\n"
		" \033[1mGeneral information                      Configuration options\033[0m\n"
		"   number of samples   {:>8}             special values                {:>3}\n"
		"   number of variables {:>8}             nominal variables             {:>3}\n"
		"   target variable     {:>8}             special handler policy {:>10}\n"
		"   target type        {:>9}             optimal grouping provided     {:>3}\n"
		"   date variable       {:>8}\n"
		"   \033[94m--------------------------------------------------------------------------\033[0m\n"
		"\n"
		" \033[1mVariables analysis                       Findings analysis (general)\033[0m\n"
		"   numerical           {:>8}             low R^2         		 {:>4}\n"
		"   ordinal             {:>8}             low corr. Pearson            {:>4}\n"
		"   categorical         {:>8}             low corr. Spearman           {:>4}\n"
		"   nominal             {:>8}             low corr. Kendall tau        {:>4}\n"
		"                                            high p-value Welch-t         {:>4}\n"
		"                                            high p-value Mann-Whitney    {:>4}\n"
		"                                            high p-value median          {:>4}\n"
		"   \033[94m--------------------------------------------------------------------------\033[0m\n"
		"\n"
		" \033[1mTiming statistics\033[0m\n"
		"   total               {:>6.3f}\n"
		"     optimal grouping  {:>6.3f} ({:>5.1%})\n"
		"       numerical         {:>6.3f} ({:>5.1%})\n"
		"       ordinal           {:>6.3f} ({:>5.1%})\n"
		"       categorical       {:>6.3f} ({:>5.1%})\n"
		"       nominal           {:>6.3f} ({:>5.1%})\n"
		"     metrics           {:>6.3f} ({:>5.1%})\n"
		"       numerical         {:>6.3f} ({:>5.1%})\n"
		"       ordinal           {:>6.3f} ({:>5.1%})\n"
		"       categorical       {:>6.3f} ({:>5.1%})\n"
		"       nominal           {:>6.3f} ({:>5.1%})\n"
		"   \033[94m--------------------------------------------------------------------------\033[0m\n"
		)
	else:
		report = (
		"\033[94m================================================================================\033[0m\n"
		"\033[1m\033[94m                      GRMlab Bivariate Continuous 0.1: Transform                           \033[0m\n"
		"\033[94m================================================================================\033[0m\n"
		"\n"
		" \033[1mResults                                        Timing statistics\033[0m\n"
		"   original data       {:>8}                   total        {:>6.3f}\n"
		"   after bivariate     {:>8}\n"
		"   removed             {:>8}\n"
		"   \033[94m--------------------------------------------------------------------------\033[0m\n"
		"\n"
		"\033[1mUser actions\033[0m\n"
		"   number of actions   {:>8}\n"
		"   number of comments  {:>8} ({:>5.1%})\n"
		"   \033[94m--------------------------------------------------------------------------\033[0m\n"
		)

	return report.format(*report_data)


class BivariateContinuous(Bivariate):
	"""
	Parameters
	----------
	target : str or None (default=None)
		The name of the variable flagged as target.

	date : str or None (default=None)
		The name of the variable flagged as date.

	optimalgrouping: object or None (default=None)
		A run instance of OptimalGrouping, performing binning of all variables
		in dataset.

	variables_nominal : list or None (default=None)
		List of ordinal variables to be treated as nominal.

	special_values : list or None (default=None)
		List of special values to be considered.

	max_pvalue_welch_t : float (default=0.05)
		The maximum allowed value for the p-value of the Welch-t test.

	max_pvalue_mann_whitney : float (default=0.05)
		The maximum allowed value for the p-value of the Mann-Whitney test.

	max_pvalue_median : float (default=0.05)
		The maximum allowed value for the p-value of the median test.

	min_corr_pearson : float (default=0.1)
		Min Pearson correlation between variable and target.

	min_corr_spearman : float (default=0.1)
		Min Spearman correlation between variable and target.

	min_corr_kendalltau : float (default=0.1)
		Min Kendall tau correlation between variable and target.

	min_r2_score : float (default=0.01)
		Min R^2 score after binning.

	monotonicity_force : boolean (default=False)
		Force ascending or descending monotonicity.

	metric : str (default="mean")
		The metric that must satisfy monotonicity constraints. Two metrics
		supported: "mean" and "median".

	special_handler_policy : str (default="join")
		Method to handle special values. Options are "join", "separate" and
		"binning". Option "join" creates an extra bucket containing all special
		values. Option "separate" creates an extra bucket for each special
		value. Option "binning" performs feature binning of special values using
		``grmlab.data_processing.feature_binning.CTree`` in order to split
		special values if these are significantly different.

	optbin_options : dict or None (default=None):
		Dictionary with options and comments to pass to a particular optbin
		instance.

	verbose : int or boolean (default=False)
		Controls verbosity of output.

	See also
	--------
	grmlab.data_processing.feature_binning.OptimalGrouping
	"""
	def __init__(self, target, date=None, optimalgrouping=None,
		variables_nominal=None, special_values=None, max_pvalue_welch_t=0.05,
		max_pvalue_mann_whitney=0.05, max_pvalue_median=0.05,
		min_corr_pearson=0.1, min_corr_spearman=0.1, min_corr_kendalltau=0.1,
		min_r2_score=0.01, monotonicity_force=False, metric="mean",
		special_handler_policy="join", optbin_options=None, verbose=False):
		
		# main input
		self.target = target
		self.date = date
		self.optimalgrouping = optimalgrouping

		# nominal variables
		self.variables_nominal = [] if variables_nominal is None else variables_nominal

		# special values options
		self.special_values = [] if special_values is None else special_values
		self.special_handler_policy = special_handler_policy

		# p-values
		self.max_pvalue_welch_t = max_pvalue_welch_t
		self.max_pvalue_mann_whitney = max_pvalue_mann_whitney
		self.max_pvalue_median = max_pvalue_median

		# correlation
		self.min_corr_pearson = min_corr_pearson
		self.min_corr_spearman = min_corr_spearman
		self.min_corr_kendalltau = min_corr_kendalltau

		# r2 score
		self.min_r2_score = min_r2_score

		# monotonicity
		self.monotonicity_force = monotonicity_force

		# metric
		self.metric = metric

		# optbin options
		self.optbin_options = optbin_options

		# others
		self.verbose = verbose

		# data samples statistics
		self._n_samples = None
		self._n_vars = None
		self._n_vars_remove = None

		# target information
		self._target = []
		self._target_dtype = None		

		# dates information
		self._dates = []
		self._unique_dates = []

		# auxiliary information
		self._column_names = []
		self._dict_variables = {}

		# timing statistics
		self._time_run = None
		self._time_run_optimalgrouping = None
		self._time_run_metrics = None
		self._time_run_numerical = 0
		self._time_run_ordinal = 0
		self._time_run_categorical = 0
		self._time_run_nominal = 0
		self._time_transform = None

		# flags
		self._is_optimalgrouping_provided = False
		self._is_run = False
		self._is_transformed = False

	def results(self, step="run", format="dataframe"):
		"""
		Return information and flags for all variables binned using OptBin.

		Parameters
		----------
		step : str or None (default="run")
			Step name, options are "run" and "transform".

		format : str, "dataframe" or "json" (default="dataframe")
			If "dataframe" return pandas.DataFrame. Otherwise, return serialized
			json.
		"""
		if not step in ("run", "transform"):
			raise ValueError("step not found.")

		if step == "run" and not self._is_run:
			raise NotRunException(self, "run")
		elif step == "transform" and not self._is_transformed:
			raise NotRunException(self, "transform")

		date = (self.date is not None)
		return _bivariate_continuous_results(self._dict_variables, step, date,
			format)

	def run(self, data):
		"""
		Run bivariate analysis.

		Note that temporal analysis is only performed if a date column is
		provided during instantiation.

		Parameters
		----------
		data : pandas.DataFrame
			Raw dataset.

		Returns
		-------
		self : object
		"""
		if not isinstance(data, pd.DataFrame):
			raise TypeError("data must be a pandas dataframe.")

		# check options
		if not isinstance(self.special_values, (list, np.ndarray)):
			raise TypeError("special values must be a list or a numpy array.")

		if not isinstance(self.variables_nominal, (list, np.ndarray)):
			raise TypeError("variables nominal must be a list or numpy array.")

		# optimal_grouping
		if self.optimalgrouping is not None:
			if not isinstance(self.optimalgrouping, OptimalGrouping):
				raise TypeError("optimalgrouping {} must be a class "
					"of type OptimalGrouping".format(
						self.optimalgrouping.__name__))
			else:
				self._is_optimalgrouping_provided = True

		self._n_samples = len(data)
		self._n_vars = len(data.columns)
		self._column_names = list(data.columns)

		# check whether given target and date are in dataframe
		if self.target is not None and not self.target in self._column_names:
			raise ValueError("target variable {} not available in dataframe."
				.format(self.target))

		if self.date is not None and not self.date in self._column_names:
			raise ValueError("date variable {} not available in dataframe."
				.format(self.date))

		# keep date data for temporal analysis
		if self.date is not None:
			self._dates = data[self.date].values
			# check date format
			check_date_format(self._dates)

			self._unique_dates = np.unique(self._dates)
			self._column_names.remove(self.date)
		else:
			print("date is not provided. Temporal analysis will not be run.")

		# target information
		self._target = data[self.target].values
		self._target_dtype = check_target_dtype(self._target)
		self._column_names.remove(self.target)

		if self._target_dtype == "binary":
			self._n_events = np.count_nonzero(self._target)
			self._n_nonevents = self._n_samples - self._n_events
			self._event_rate = self._n_events / self._n_samples

		# check optimalgrouping parameters
		if self.optimalgrouping is not None:
			if self.optimalgrouping._target_name != self._target_name:
				raise ValueError("target ({}) and optimalgrouping "
					"target ({}) must coincide.".format(
						self._target_name, self.optimalgrouping._target_name))

			if self.optimalgrouping._is_run:
				if self._target_dtype != self.optimalgrouping._target_dtype:
					raise ValueError("target dtype and optimalgrouping "
						"dtype must coincide. {} != {}".format(
							self._target_dtype,
							self.optimalgrouping._target_dtype))

				if self.optimalgrouping._n_samples != self._n_samples:
					raise ValueError("target and optimalgrouping target "
						"dimensions must coincide. {} != {}".format(
							self._n_samples, self.optimalgrouping._n_samples))

				if any(self.optimalgrouping._target != self._target):
					raise ValueError("target and optimalgrouping target must "
						"coincide.")

		# run bivariate
		self._run(data)

		self._is_run = True

		return self

	def stats(self, step="run"):
		"""
		Bivariate analysis statistics.

		Parameters
		----------
		step : str or None (default="run")
			Step name, options are "run" and "transform".
		"""
		if not step in ("run", "transform"):
			raise ValueError("step not found.")

		if not self._is_run and step == "run":
			raise NotRunException(self, "run")
		elif not self._is_transformed and step == "transform":
			raise NotRunException(self, "transform")

		dict_values = self._dict_variables.values()

		if step == "run":
			special_flag = "yes" if self.special_values else "no"
			nominal_flag = "yes" if self.variables_nominal else "no"
			date_flag = self.date if self.date is not None else "not set"
			optimal_grouping_flag = "yes" if self._is_optimalgrouping_provided else "no"

			# timing
			perc_time_numerical = self._time_run_numerical / self._time_run
			perc_time_ordinal = self._time_run_ordinal / self._time_run
			perc_time_categorical = self._time_run_categorical / self._time_run
			perc_time_nominal = self._time_run_nominal / self._time_run

			perc_time_og_numerical = (self.optimalgrouping._time_run_numerical
				/ self.optimalgrouping._time_run)
			perc_time_og_ordinal = (self.optimalgrouping._time_run_ordinal
				/ self.optimalgrouping._time_run)
			perc_time_og_categorical = (
				self.optimalgrouping._time_run_categorical
				/ self.optimalgrouping._time_run)
			perc_time_og_nominal = (self.optimalgrouping._time_run_nominal
				/ self.optimalgrouping._time_run)

			perc_time_og = self.optimalgrouping._time_run / self._time_run
			time_metrics = self._time_run - self.optimalgrouping._time_run
			perc_time_metrics = time_metrics / self._time_run

			[n_numerical, n_ordinal, n_categorical, n_nominal] = [
				sum(d["dtype"] == dtype for d in dict_values) for dtype
				in ("numerical", "ordinal", "categorical", "nominal")]

			[pvalue_welch_t_high, pvalue_mann_whitney_high, pvalue_median_high,
				r2_score_low, undefined, corr_pearson_low, corr_kendalltau_low,
				corr_spearman_low] = [
					sum(status in d["status"] for d in dict_values) for status
					in STATUS_REMOVE + STATUS_REVIEW]

			# prepare data
			report_data = [self._n_samples, special_flag, self._n_vars,
				nominal_flag, self.target, self.special_handler_policy,
				self._target_dtype, optimal_grouping_flag, date_flag,
				n_numerical, r2_score_low, n_ordinal, corr_pearson_low, n_categorical,
				corr_spearman_low, n_nominal, corr_kendalltau_low,
				pvalue_welch_t_high, pvalue_mann_whitney_high,
				pvalue_median_high, self._time_run, self.optimalgrouping._time_run,
				perc_time_og, self.optimalgrouping._time_run_numerical,
				perc_time_og_numerical, self.optimalgrouping._time_run_ordinal,
				perc_time_og_ordinal, self.optimalgrouping._time_run_categorical,
				perc_time_og_categorical, self.optimalgrouping._time_run_nominal,
				perc_time_og_nominal, time_metrics, perc_time_metrics,
				self._time_run_numerical, perc_time_numerical,
				self._time_run_ordinal, perc_time_ordinal,
				self._time_run_categorical, perc_time_categorical,
				self._time_run_nominal, perc_time_nominal]
		else:
			n_vars_after = self._n_vars - self._n_vars_remove

			n_user_actions = sum(1 for d in dict_values if "user_action" in d)
			n_user_comment = sum(d["user_comment"] != "" for d in dict_values
				if "user_action" in d)

			if n_user_actions:
				perc_user_comment = n_user_comment / n_user_actions
			else:
				perc_user_comment = 0

			report_data = [self._n_vars, self._time_transform, n_vars_after,
				self._n_vars_remove, n_user_actions, n_user_comment,
				perc_user_comment]

		print(_bivariate_continuous_stats(report_data, step))

	def variable_stats_descriptive(self, variable):
		"""
		Generate a detailed descriptive analysis report for a given variable.

		Parameters
		----------
		variable : str
			The variable name.
		"""
		if not self._is_run:
			raise NotRunException(self, "run")

		if variable not in self._column_names:
			raise ValueError("variable '{}' not in data.".format(variable))

		data = self._dict_variables[variable]

		if data["dtype"] in ("numerical", "ordinal"):
			print(_variable_descriptive_numerical(variable, data))
		else:
			print(_variable_descriptive_categorical(variable, data))

	def variable_stats_temporal(self, variable):
		"""
		Generate a detailed temporal analysis report for a given variable.

		Parameters
		----------
		name : str
			The variable name.		
		"""
		if not self._is_run:
			raise NotRunException(self, "run")

		if variable not in self._column_names:
			raise ValueError("variable '{}' not in data.".format(variable))

		if self.date is None:
			raise ValueError("date is not provided. Temporal analysis was "
				"not run.")

		print(_variable_temporal(variable, self._dict_variables[variable]))

	def variable_plot_descriptive(self, variable, metric=None):
		"""
		Plot optimal binning for a given variable.

		Parameters
		----------
		variable : str
			The variable name.

		metric : str (default="mean")
			The metric to plot. Two metrics supported: "mean" and "median".
			If None, the metric selected to satisfy monotonicity constraints
			is plotted.
		"""
		if not self._is_run:
			raise NotRunException(self, "run")

		if variable not in self._column_names:
			raise ValueError("variable '{}' not in data.".format(variable))

		optbin = self._get_optbin_variable(variable)
		basic_binning_table = self._dict_variables[variable]["binning_table"]
		basic_splits = self._dict_variables[variable]["splits"]

		_variable_plot_descriptive(optbin, basic_binning_table, basic_splits,
			metric)

	def variable_plot_temporal(self, variable, metric="mean"):
		"""
		Plot temporal analysis for a given variable.

		Parameters
		----------
		variable : str
			The variable name.

		metric : str (default="mean")
			The metric to plot. Two metrics supported: "mean" and "median".
		"""
		if not self._is_run:
			raise NotRunException(self, "run")

		if variable not in self._column_names:
			raise ValueError("variable '{}' not in data.".format(variable))

		if self.date is None:
			raise ValueError("date is not provided. Temporal analysis was "
				"not run.")

		if metric not in ("mean", "median"):
			raise ValueError("metric {} not supported.".format(metric))

		_variable_plot_temporal(variable, self._dict_variables[variable],
			metric)

	def _run_optimal_grouping(self, data, sample_weight=None):
		"""Run optimal grouping if needed."""
		time_init = time.perf_counter()

		# if optimalgrouping is provided => check status. Run is needed.
		if self.optimalgrouping is None or not self.optimalgrouping._is_run:
			self.optimalgrouping = OptimalGrouping(target=self.target,
				variables_nominal=self.variables_nominal,
				special_values=self.special_values,
				monotonicity_force=self.monotonicity_force,
				continuous_metric=self.metric,
				special_handler_policy=self.special_handler_policy,
				optbin_options=self.optbin_options,
				verbose=self.verbose)

			# run optimalgrouping and catch exceptions
			try:
				self.optimalgrouping.run(data)
			except EnvironmentError as err:
				print("optimalgrouping - unexpected error: {}".format(
					sys.exc_info()[0]))
			except Exception as err:
				print("optimalgrouping - error: {}".format(err))
		else:
			if self.verbose:
				print("optimalgrouping: terminated successfully.")

		self._time_run_optimalgrouping = time.perf_counter() - time_init

	def _run_variable_numerical(self, data, name, dtype, sample_weight=None):
		idx_missing = data.isna()

		if self.special_values:
			idx_special = data.isin(self.special_values)

		# clean data to compute descriptive statistics
		if self.special_values:
			idx_clean = (~idx_special & ~idx_missing)
		else:
			idx_clean = ~idx_missing

		data_clean = data[idx_clean]
		n_informed = len(data_clean)

		if dtype == "numerical" and n_informed < self._n_samples:
			data_int = data_clean.astype(np.int)
			if all(data_clean == data_int):
				data_clean = data_int
				dtype = "ordinal"

				if self.verbose:
					print("datatype check: variable was casted as ordinal.")

		# generate binning table information using simple buckets
		percentiles_20 = np.linspace(5, 100, 20).astype(np.int)
		splits = np.percentile(data_clean, percentiles_20)

		try:
			binning_table = table(values=data.values, target=self._target,
				splits=splits, special_values=self.special_values)[0]
		except Exception as err:
			print("Binning table for variable {} was not computed."
				"Error: {}.".format(name, str(err)))		

		# get optbin binning table
		optbin = self._get_optbin_variable(name)
		optbin_binning_table = optbin.binning_table()

		# calculate p-value from various significance tests
		tb = optbin_binning_table.iloc[:, [1, 4, 6]
			].values[:optbin._n_optimal_buckets]

		[pvalue_welch_t, pvalue_mann_whitney, pvalue_median] = _compute_pvalues(
			data.values, self._target, optbin.splits_optimal,
			self.special_values, tb)

		# transform data
		xt = optbin.transform(data.values)

		# calculate correlation
		corr_pearson = pearsonr(xt, self._target)[0]
		corr_spearman = spearmanr(xt, self._target)[0]
		corr_kendalltau = kendalltau(xt, self._target)[0]

		corr_spearman = 0 if np.isnan(corr_spearman) else corr_spearman
		corr_kendalltau = 0 if np.isnan(corr_kendalltau) else corr_kendalltau

		# fitting statistics
		mae = mean_absolute_error(self._target, xt)
		mse = mean_squared_error(self._target, xt)
		mdae = median_absolute_error(self._target, xt)
		r2 = r2_score(self._target, xt)

		# variable dict
		fields = ["dtype", "pvalue_welch_t", "pvalue_mann_whitney",
			"pvalue_median", "corr_pearson", "corr_spearman", "corr_kendalltau",
			"monotonicity", "groups", "group_special", "group_missing",
			"group_others", "optimal_binning_table", "splits", "binning_table",
			"mean_absolute_error", "mean_squared_error", "median_absolute_error",
			"r2_score"]

		info = [dtype, pvalue_welch_t, pvalue_mann_whitney, pvalue_median,
			corr_pearson, corr_spearman, corr_kendalltau,
			optbin.monotonicity_sense, optbin._n_optimal_buckets,
			optbin._group_special, optbin._group_missing, optbin._group_others,
			optbin_binning_table, splits, binning_table, mae, mse, mdae, r2]

		dict_variable_info = dict(zip(fields, info))

		if self.date is not None:
			dict_variable_t_info = self._run_temporal(data, name, dtype)
			dict_variable_info = {**dict_variable_info, **dict_variable_t_info}

		return dict_variable_info

	def _run_variable_categorical(self, data, name, dtype, sample_weight=None):
		idx_missing = data.isna()

		if self.special_values:
			idx_special = data.isin(self.special_values)

		# clean data to compute descriptive statistics
		if self.special_values:
			idx_clean = (~idx_special & ~idx_missing)
		else:
			idx_clean = ~idx_missingpd

		data_clean = data[idx_clean]
		n_informed = len(data_clean)

		if dtype == "categorical":
			data_clean = data_clean.astype(str)

		# number of categories
		unique_categories = data_clean.value_counts()
		u_categories = unique_categories.index.values
		n_u_categories = unique_categories.values
		n_categories = len(u_categories)
		most_freq_category = u_categories[0]
		p_most_freq_category = n_u_categories[0] / n_informed

		# top 10 most frequent categories
		max_categories = min(n_categories, 10)
		top_categories = u_categories[:max_categories]
		n_top_categories = n_u_categories[:max_categories]

		others = [c for c in u_categories if c not in top_categories]
		splits = [[c] for c in top_categories] + [others]

		# generate binning table with top categories and others
		try:
			binning_table = table(values=data, target=self._target,
				splits=splits, special_values=self.special_values)[0]
		except Exception as err:
			print("Binning table for variable {} was not computed."
				"Error: {}.".format(name, str(err)))

		# get optbin binning table
		optbin = self._get_optbin_variable(name)
		optbin_binning_table = optbin.binning_table()

		# distribution of number of categories
		n_categories_split = np.asarray([len(x) for x in optbin.splits_optimal])

		if n_categories_split.size:
			mean_n_categories = np.mean(n_categories_split)
			std_n_categories = np.std(n_categories_split)
			min_n_categories = np.min(n_categories_split)
			max_n_categories = np.max(n_categories_split)

			[percentile_25_n_categories, median_n_categories,
				percentile_75_n_categories] = np.percentile(n_categories_split,
					[25, 50, 75])
		else:
			[mean_n_categories, std_n_categories, min_n_categories,
				max_n_categories, percentile_25_n_categories,
				median_n_categories, percentile_75_n_categories] = [np.nan] * 7

		# calculate p-value from various significance tests
		tb = optbin_binning_table.iloc[:, [1, 4, 6]
			].values[:optbin._n_optimal_buckets]

		# calculate p-value from various significance tests
		[pvalue_welch_t, pvalue_mann_whitney, pvalue_median] = _compute_pvalues(
			data.values, self._target, optbin.splits_optimal,
			self.special_values, tb)

		# transform data
		xt = optbin.transform(data.values)

		# calculate correlation
		corr_pearson = pearsonr(xt, self._target)[0]
		corr_spearman = spearmanr(xt, self._target)[0]
		corr_kendalltau = kendalltau(xt, self._target)[0]

		corr_spearman = 0 if np.isnan(corr_spearman) else corr_spearman
		corr_kendalltau = 0 if np.isnan(corr_kendalltau) else corr_kendalltau

		# fitting statistics
		mae = mean_absolute_error(self._target, xt)
		mse = mean_squared_error(self._target, xt)
		mdae = median_absolute_error(self._target, xt)
		r2 = r2_score(self._target, xt)

		# variable dict
		fields = ["dtype", "pvalue_welch_t", "pvalue_mann_whitney",
			"pvalue_median", "corr_pearson", "corr_spearman", "corr_kendalltau",
			"categories", "n_categories", "most_freq_category",
			"p_most_freq_category", "top_categories", "n_top_categories",
			"mean_n_categories", "std_n_categories", "min_n_categories",
			"percentile_25_n_categories", "median_n_categories",
			"percentile_75_n_categories", "max_n_categories", "monotonicity",
			"groups", "group_special", "group_missing", "group_others",
			"optimal_binning_table", "splits", "binning_table",
			"mean_absolute_error", "mean_squared_error", "median_absolute_error",
			"r2_score"]

		info = [dtype, pvalue_welch_t, pvalue_mann_whitney, pvalue_median,
			corr_pearson, corr_spearman, corr_kendalltau, u_categories,
			n_categories, most_freq_category, p_most_freq_category,
			top_categories, n_top_categories, mean_n_categories,
			std_n_categories, min_n_categories, percentile_25_n_categories,
			median_n_categories, percentile_75_n_categories, max_n_categories,
			optbin.monotonicity_sense, optbin._n_optimal_buckets,
			optbin._group_special, optbin._group_missing, optbin._group_others,
			optbin_binning_table, splits, binning_table, mae, mse, mdae, r2]

		dict_variable_info = dict(zip(fields, info))

		if self.date is not None:
			dict_variable_t_info = self._run_temporal(data, name, dtype)
			dict_variable_info = {**dict_variable_info, **dict_variable_t_info}

		return dict_variable_info

	def _run_temporal(self, data, name, dtype):
		# get optbin variable
		optbin = self._get_optbin_variable(name)

		# optimal splits
		optbin_splits = optbin.splits_optimal
		n_splits = len(optbin_splits)

		# initialize woe and groups arrays
		t_split = np.zeros(self._n_samples)

		if not n_splits:
			t_split = np.zeros(self._n_samples)
			idx_nan = data.isna()
			# a single category increase n_splits for special and missing
			n_splits = 1
		elif dtype in ("categorical", "nominal"):
			# categorical and nominal variables return groups as 
			# numpy.ndarray objects.
			for _idx ,split in enumerate(optbin_splits):
				mask = data.isin(split).values
				t_split[mask] = _idx			
			idx_nan = data.isna()
		else:
			# numerical and ordinal variables return extra group
			# (> last split)
			splits = optbin_splits[::-1]
			x = data.values
			mask = (x > splits[-1])
			t_split[mask] = n_splits
			for _idx, split in enumerate(splits):
				mask = (x <= split)
				t_split[mask] = n_splits - (_idx + 1)
			# indexes with NaN n x
			idx_nan = data.isna()
			# account for > group
			n_splits += 1

		# special values
		if optbin._splits_specials:
			x = data.values
			for _idx, split in enumerate(optbin._splits_specials):
				if isinstance(split, np.ndarray):
					mask = data.isin(split).values
				else:
					mask = (x == split)
				t_split[mask] = n_splits
				n_splits += 1
		else:
			n_splits += 1

		# missing values
		t_split[idx_nan] = n_splits

		# filter by unique date and store occurrences and event information
		n_unique_dates = len(self._unique_dates)
		t_split_count_by_date = np.zeros((n_splits+1, n_unique_dates))
		t_split_mean_by_date = np.zeros((n_splits+1, n_unique_dates))
		t_split_median_by_date = np.zeros((n_splits+1, n_unique_dates))

		t_r2_by_date = np.zeros(n_unique_dates)
		xt = optbin.transform(data.values)

		for id_date, udate in enumerate(self._unique_dates):
			mask_date = (self._dates == udate)

			t_r2_by_date[id_date] = r2_score(self._target[mask_date],
				xt[mask_date])
		
			for split in range(n_splits + 1):
				mask_split_date = ((t_split == split) & mask_date)
				t_split_count_by_date[split, id_date] = np.count_nonzero(
					mask_split_date)				
				t_split_mean_by_date[split, id_date] = np.mean(
					self._target[mask_split_date])
				t_split_median_by_date[split, id_date] = np.median(
					self._target[mask_split_date])

		t_n_split_empty = np.count_nonzero(t_split_count_by_date == 0, axis=0)

		# mean temporal analysis
		t_mean_mean = np.mean(t_split_mean_by_date, axis=1)
		t_mean_std = np.std(t_split_mean_by_date, axis=1)
		t_mean_min = np.min(t_split_mean_by_date, axis=1)
		t_mean_max = np.max(t_split_mean_by_date, axis=1)
		[t_mean_percentile_25, t_mean_median,
			t_mean_percentile_75] = np.percentile(t_split_mean_by_date,
				[25, 50, 75], axis=1)
		iqr_n = t_mean_percentile_75 - t_mean_percentile_25
		iqr_d = t_mean_percentile_75 + t_mean_percentile_25
		t_mean_dispersion = iqr_n / iqr_d

		# median temporal analysis
		t_median_mean = np.mean(t_split_median_by_date, axis=1)
		t_median_std = np.std(t_split_median_by_date, axis=1)
		t_median_min = np.min(t_split_median_by_date, axis=1)
		t_median_max = np.max(t_split_median_by_date, axis=1)
		[t_median_percentile_25, t_median_median,
			t_median_percentile_75] = np.percentile(t_split_median_by_date,
				[25, 50, 75], axis=1)

		iqr_n = t_median_percentile_75 - t_median_percentile_25
		iqr_d = t_median_percentile_75 + t_median_percentile_25
		t_median_dispersion = iqr_n / iqr_d

		# R^2
		t_r2_mean = np.nanmean(t_r2_by_date)
		t_r2_std = np.nanstd(t_r2_by_date)
		t_r2_min = np.nanmin(t_r2_by_date)
		t_r2_max = np.nanmax(t_r2_by_date)
		[t_r2_percentile_25, t_r2_median,
			t_r2_percentile_75] = np.nanpercentile(t_r2_by_date, [25, 50, 75])

		# divergence
		split_date = t_split_count_by_date / np.sum(t_split_count_by_date,
			axis=0)
		t_split_div_uniform = js_divergence_multivariate(split_date, "uniform")
		t_split_div_exponential = js_divergence_multivariate(split_date,
			"exponential")

		fields = ["t_split_count_by_date", "t_split_mean_by_date",
			"t_split_median_by_date", "t_n_split_empty", "t_mean_mean",
			"t_mean_std", "t_mean_min", "t_mean_percentile_25", "t_mean_median",
			"t_mean_percentile_75", "t_mean_max", "t_mean_dispersion",
			"t_median_mean", "t_median_std", "t_median_min",
			"t_median_percentile_25", "t_median_median",
			"t_median_percentile_75", "t_median_max", "t_median_dispersion",
			"t_r2_mean", "t_r2_std", "t_r2_min", "t_r2_percentile_25",
			"t_r2_median", "t_r2_percentile_75", "t_r2_max",
			"t_split_div_uniform", "t_split_div_exponential"]

		info = [t_split_count_by_date, t_split_mean_by_date,
			t_split_median_by_date, t_n_split_empty, t_mean_mean,
			t_mean_std, t_mean_min, t_mean_percentile_25, t_mean_median,
			t_mean_percentile_75, t_mean_max, t_mean_dispersion,
			t_median_mean, t_median_std, t_median_min,
			t_median_percentile_25, t_median_median,
			t_median_percentile_75, t_median_max, t_median_dispersion,
			t_r2_mean, t_r2_std, t_r2_min, t_r2_percentile_25,
			t_r2_median, t_r2_percentile_75, t_r2_max,
			t_split_div_uniform, t_split_div_exponential]

		return dict(zip(fields, info))

	def _decision_engine(self):
		"""Set action to be taken for each variable."""
		MSG_METRIC_HIGH = "{} value above threshold {:.3f} > {:.3f}"
		MSG_METRIC_LOW = "{} value below threshold {:.3f} < {:.3f}"
		MSG_UNDEFINED = "undefined monotonicity"

		for column in self._column_names:

			# initialize statuses and comments
			self._dict_variables[column]["status"] = []
			self._dict_variables[column]["comment"] = []

			# correlations
			if (abs(self._dict_variables[column]["corr_kendalltau"])
				< self.min_corr_kendalltau):
				self._dict_variables[column]["status"].append(
					STATUS_CORR_KENDALLTAU_LOW)
				self._dict_variables[column]["comment"].append(
					MSG_METRIC_LOW.format("corr_kendalltau",
						self._dict_variables[column]["corr_kendalltau"],
						self.min_corr_kendalltau))

			if (abs(self._dict_variables[column]["corr_spearman"])
				< self.min_corr_kendalltau):
				self._dict_variables[column]["status"].append(
					STATUS_CORR_SPEARMAN_LOW)
				self._dict_variables[column]["comment"].append(
					MSG_METRIC_LOW.format("corr_spearman",
						abs(self._dict_variables[column]["corr_spearman"]),
						self.min_corr_spearman))

			# p-values
			if (self._dict_variables[column]["pvalue_welch_t"]
				> self.max_pvalue_welch_t):
				self._dict_variables[column]["status"].append(
					STATUS_PVALUE_WELCH_T_HIGH)
				self._dict_variables[column]["comment"].append(
					MSG_METRIC_HIGH.format("pvalue_welch_t",
						self._dict_variables[column]["pvalue_welch_t"],
						self.max_pvalue_welch_t))
				
			if (self._dict_variables[column]["pvalue_mann_whitney"]
				> self.max_pvalue_mann_whitney):
				self._dict_variables[column]["status"].append(
					STATUS_PVALUE_MANN_WHITNEY_HIGH)
				self._dict_variables[column]["comment"].append(
					MSG_METRIC_HIGH.format("pvalue_mann_whitney",
						self._dict_variables[column]["pvalue_mann_whitney"],
						self.max_pvalue_mann_whitney))

			if (self._dict_variables[column]["pvalue_median"]
				> self.max_pvalue_median):
				self._dict_variables[column]["status"].append(
					STATUS_PVALUE_MEDIAN_HIGH)
				self._dict_variables[column]["comment"].append(
					MSG_METRIC_HIGH.format("pvalue_median",
					self._dict_variables[column]["pvalue_median"],
					self.max_pvalue_median))

			# optimalgrouping undefined
			optbin = self._get_optbin_variable(column)
			if optbin.monotonicity_sense == "undefined":
				self._dict_variables[column]["status"].append(STATUS_UNDEFINED)
				self._dict_variables[column]["comment"].append(MSG_UNDEFINED)

			if self._dict_variables[column]["dtype"] in ("numerical",
				"ordinal"):

				# correlations
				if (abs(self._dict_variables[column]["corr_pearson"])
					< self.min_corr_pearson):
					self._dict_variables[column]["status"].append(
						STATUS_CORR_PEARSON_LOW)
					self._dict_variables[column]["comment"].append(
						MSG_METRIC_LOW.format("corr_pearson",
							abs(self._dict_variables[column]["corr_pearson"]),
							self.min_corr_pearson))

				# R^2 score
				if self._dict_variables[column]["r2_score"] < self.min_r2_score:
					self._dict_variables[column]["status"].append(
						STATUS_R2_SCORE_LOW)
					self._dict_variables[column]["comment"].append(
						MSG_METRIC_LOW.format("r2_score",
							abs(self._dict_variables[column]["r2_score"]),
							self.min_r2_score))

			status = self._dict_variables[column]["status"]
			if any(st in STATUS_REMOVE for st in status):
				self._dict_variables[column]["action"] = "remove"
			elif any(st in STATUS_REVIEW for st in status):
				self._dict_variables[column]["action"] = "review"
			else:
				self._dict_variables[column]["action"] = "keep"

			action = self._dict_variables[column]["action"]
			self._dict_variables[column]["recommended_action"] = action

			if self.optbin_options is not None:
				if column in self.optbin_options.keys():
					# pass optbin options
					user_params = self.optbin_options[column]["params"]
					user_comment = self.optbin_options[column]["comment"]

					self._dict_variables[column]["optbin_params"] = user_params
					self._dict_variables[column]["optbin_comment"] = user_comment
					
					# pass optbin options differences
					optbin_default = self.optimalgrouping._dict_optbin_options[column]
					self._dict_variables[column]["optbin_default"] = optbin_default
