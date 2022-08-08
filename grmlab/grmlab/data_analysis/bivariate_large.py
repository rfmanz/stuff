"""
Bivariate analysis by columns.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..core.dtypes import check_date_format
from ..core.dtypes import check_target_dtype
from ..data_processing.feature_binning import OptimalGroupingByColumns
from .bivariate import Bivariate


class BivariateByColumns(Bivariate):
	"""
	Parameters
	----------
	target : str or None (default=None)
		The name of the variable flagged as target.

	date : str or None (default=None)
		The name of the variable flagged as date.

	variables_nominal : list or None (default=None)
		List of ordinal variables to be treated as nominal.

	special_values : list or None (default=None)
		List of special values to be considered.

	min_iv : float (default=0.015)
		The minimum allowed value for the Information Value (IV) metric.

	max_iv : float (default=0.7)
		The maximum allowed value for the Information Value (IV) metric.

	min_gini : float (default=0.025)
		The minimum allowed value for the Gini metric.

	max_gini : float (default=0.85)
		The maximum allowed value for the Gini metric.

	max_pvalue_anova : float (default=0.05)
		The maximum allowed value for the p-value of the Anova test.

	max_pvalue_chi2 : float (default=0.05)
		The maximum allowed value for the p-value of the Chi2 test.

	max_pvalue_ks : float (default=0.05)
		The maximum allowed value for the p-value of the Kolmogorov-Smirnov test.

	max_pvalue_kw : float (default=0.05)
		The maximum allowed value for the p-value of the Kruskal-Wallis test.

	max_pvalue_median_t : float (default=0.05)
		The maximum allowed value for the p-value of the Median-t test.

	min_cramer_v : float (default=0.1)
		The minimum allowed value for the measure of association Cramer's V.

	special_handler_policy : str (default="join")
		Method to handle special values. Options are "join", "separate" and
		"binning". Option "join" creates an extra bucket containing all special
		values. Option "separate" creates an extra bucket for each special
		value. Option "binning" performs feature binning of special values using
		``grmlab.data_processing.feature_binning.CTree`` in order to split
		special values if these are significantly different.

	special_woe_policy : str (default=empirical)
		Weight-of-Evidence (WoE) value to be assigned to special values buckets.
		Options supported are: "empirical", "worst", "zero". Option "empirical"
		assign the actual WoE value. Option "worst" assigns the WoE value
		corresponding to the bucket with the highest event rate. Finally, option
		"zero" assigns value 0.

	missing_woe_policy : str (default="empirical")
		Weight-of-Evidence (WoE) value to be assigned to missing values bucket.
		Options supported are: "empirical", "worst", "zero". Option "empirical"
		assign the actual WoE value. Option "worst" assigns the WoE value
		corresponding to the bucket with the highest event rate. Finally, option
		"zero" assigns value 0.

	verbose : int or boolean (default=False)
		Controls verbosity of output.

	See also
	--------
	grmlab.data_processing.feature_binning.OptimalGroupingByColumns
	"""	
	def __init__(self, target, date=None, variables_nominal=None,
		special_values=None, min_iv=0.015, max_iv=0.7, min_gini=0.025,
		max_gini=0.85, max_pvalue_anova=0.05, max_pvalue_chi2=0.05,
		max_pvalue_ks=0.05, max_pvalue_kw=0.05, max_pvalue_median_t=0.05,
		min_cramer_v=0.1, special_handler_policy="join",
		special_woe_policy="empirical", missing_woe_policy="empirical",
		verbose=False):

		super().__init__(target, date, None, variables_nominal,
			special_values, min_iv, max_iv, min_gini, max_gini,
			max_pvalue_anova, max_pvalue_chi2, max_pvalue_ks, max_pvalue_kw,
			max_pvalue_median_t, min_cramer_v, special_handler_policy,
			special_woe_policy, missing_woe_policy, verbose)

		self.path_csv = None
		self.sep_csv = None
		self.path_parquet = None

	def run(self, path, format, sep_csv=","):
		"""
		Run bivariate data analysis reading one column at a time.

		Parameters
		----------
		path : str
			The string path to a CSV or Parquet file.

		format : str
			The file format. Currently only 'csv' and 'parquet' options are
			supported.

		sep_csv : str (default=",")
			Delimiter / separator to use when reading a CSV file.

		Returns
		-------
		self : object
		"""		
		if not format in ("csv", "parquet"):
			raise ValueError("file format {} is not supported.".format(format))

		if format == "csv":
			self.path_csv = path
			self.sep_csv = sep_csv
			self._setup_csv()
		elif format == "parquet":
			self.path_parquet = path
			self._setup_parquet()

		# run bivariate
		self._run(file_format=format)

	def transform(self):
		"""
		Transform input dataset in-place.

		Reduce the raw dataset by removing columns with action flag equal to
		remove.

		Parameters
		----------
		data : pandas.DataFrame
			Raw dataset.

		mode : str
			Transformation mode, options are "aggresive" and "basic". If
			``mode=="aggressive"`` variables tagged with action "remove" and
			"review" are dropped. If ``mode=="basic"`` only variables tagged as
			"remove" are dropped.

		Returns
		-------
		self : object

		Warning
		-------
		Currently not implemented.
		"""
		raise NotImplementedError

	def _read_column_csv(self, column):
		"""Read an specific column from CSV file."""
		return pd.read_csv(filepath_or_buffer=self.path_csv, sep=self.sep_csv,
			engine='c', encoding='latin-1', low_memory=False, memory_map=True,
			usecols=[column]).iloc[:, 0]

	def _read_column_parquet(self, column):
		"""Read an specific column from Parquet file."""
		return pq.read_table(source=self.path_parquet,
			columns=[column]).to_pandas().iloc[:, 0]

	def _read_header_csv(self):
		"""Read CSV header."""
		return list(pd.read_csv(filepath_or_buffer=self.path_csv,
			sep=self.sep_csv, engine='c', encoding='latin-1', low_memory=False,
			memory_map=True, header=None, nrows=1).values[0])

	def _run(self, file_format):
		"""Run optimalgrouping, bivariate and decision engine."""
		time_init = time.perf_counter()

		# run optimalgrouping
		self._run_optimal_grouping(file_format)

		if self.verbose:
			print("running bivariate analysis...")

		for id, name in enumerate(self._column_names):
			if self.verbose:
				print("\nvariable {}: {}".format(id, name))

			if file_format == "csv":
				self._run_variable(self._read_column_csv(name), name)
			elif file_format == "parquet":
				self._run_variable(self._read_column_parquet(name), name)

		# set actions
		self._decision_engine()			

		self._time_run = time.perf_counter() - time_init

		# update flag
		self._is_run = True

		return self

	def _run_optimal_grouping(self, file_format):
		"""Run optimal grouping by columns"""
		self.optimalgrouping = OptimalGroupingByColumns(self.target,
			self.variables_nominal, self.special_values,
			self.special_handler_policy, self.special_woe_policy,
			self.missing_woe_policy, self.verbose)

		if file_format == "csv":
			self.optimalgrouping.run(self.path_csv, file_format, self.sep_csv)
		else:
			self.optimalgrouping.run(self.path_parquet, file_format)

		self._time_run_optimalgrouping = self.optimalgrouping._time_run

	def _setup_checks(self):
		# special values
		if not isinstance(self.special_values, (list, np.ndarray)):
			raise TypeError("special_values must be a list or numpy array.")

		# special_handler_policy
		if not self.special_handler_policy in ("join", "separate", "binning"):
			raise ValueError("special_handler_policy option not supported.")

		# special_woe_policy
		if not self.special_woe_policy in ("empirical", "worst", "zero"):
			raise ValueError("special_woe_policy option not supported.")

		# missing_woe_policy
		if not self.missing_woe_policy in ("empirical", "worst", "zero"):
			raise ValueError("missing_woe_policy option not supported.")

	def _setup_csv(self):
		"""
		Perform preliminary analysis of the input dataset and check target
		column.
		"""
		self._column_names = self._read_header_csv()
		self._n_vars = len(self._column_names)

		if not self.target in self._column_names:
			raise ValueError("target column {} not available in dataframe."
				.format(self.target))

		if self.date is not None and not self.date in self._column_names:
			raise ValueError("date column {} not available in dataframe."
				.format(self.date))

		# select target data. save target to avoid querying several times.
		self._target = self._read_column_csv(self.target).values

		# check target type
		self._target_dtype = check_target_dtype(self._target)

		if self.verbose:
			print("target type is {}.".format(self._target_dtype))

		if self.date is not None:
			self._dates = self._read_column_parquet(self.date).values
			check_date_format(self._dates)

			self._unique_dates = np.unique(self._dates)
			self._n_samples = len(self._dates)
			self._column_names.remove(self.date)

		# exclude target from the list of variables
		self._column_names.remove(self.target)

		self._n_samples= len(self._target)

	def _setup_parquet(self):
		"""
		Perform preliminary analysis of the input dataset and check target
		column.
		"""
		file = pq.ParquetFile(source=self.path_parquet)
		self._column_names = file.schema.names
		self._n_samples = file.metadata.num_rows
		self._n_vars = len(self._column_names)

		if not self.target in self._column_names:
			raise ValueError("target column {} not available in dataframe."
				.format(self.target))

		if self.date is not None and not self.date in self._column_names:
			raise ValueError("date column {} not available in dataframe."
				.format(self.date))

		# select target data. save target to avoid querying several times.
		self._target = self._read_column_parquet(self.target).values

		# check target type
		self._target_dtype = check_target_dtype(self._target)

		if self._target_dtype == "binary":
			self._n_events = np.count_nonzero(self._target)
			self._n_nonevents = self._n_samples - self._n_events
			self._event_rate = self._n_events / self._n_samples

		if self.verbose:
			print("target type is {}.".format(self._target_dtype))

		if self.date is not None:
			self._dates = self._read_column_parquet(self.date).values
			check_date_format(self._dates)

			self._unique_dates = np.unique(self._dates)
			self._column_names.remove(self.date)

		# exclude target from the list of variables
		self._column_names.remove(self.target)
