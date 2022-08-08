"""
Optimal Grouping automatic process for large datasets.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2018.

import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ...core.dtypes import check_dtype
from ...core.dtypes import check_target_dtype
from ...core.exceptions import NotRunException
from .optbin import OptBin
from .optbin_continuous import OptBinContinuous
from .optimalgrouping import OptimalGrouping


class OptimalGroupingByColumns(OptimalGrouping):
	"""
	Automatic optimal grouping procedure to perform binning to all variables
	in dataset.

	Parameters
	----------
	target : str
		The name of the target variable.

	variables_nominal : list or None (default=None)
		List of ordinal variables to be treated as nominal.

	special_values : list or None (default=None)
		List of special values to be considered.

	monotonicity_force : boolean (default=False)
		Force ascending or descending monotonicity.

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

	See also
	--------
	CTree, OptBin, OptBinContinuous

	Notes
	-----
	This procedure calls ``grmlab.data_processing.feature_binning.OptBin`` or
	``grmlab.data_processing.feature_binning.OptBinContinuous`` for each
	variable in the dataset. Different prebinning algorithms are considered
	depending on the variable and target data type.
	"""
	def __init__(self, target, variables_nominal=None, special_values=None,
		monotonicity_force=False, special_handler_policy="join",
		special_woe_policy="empirical", missing_woe_policy="empirical",
		verbose=False):

		super().__init__(target, variables_nominal, special_values,
			monotonicity_force, special_handler_policy, special_woe_policy,
			missing_woe_policy, verbose)

		self.path_csv = None
		self.sep_csv = None
		self.path_parquet = None

	def run(self, path, format, sep_csv=","):
		"""
		Run optimal automatic grouping reading one column at a time.

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

		self._setup_checks()

		if format == "csv":
			self.path_csv = path
			self.sep_csv = sep_csv
			self._setup_csv()
		elif format == "parquet":
			self.path_parquet = path
			self._setup_parquet()

		# run optimal grouping
		self._run(file_format=format)

	def transform(self):
		"""
		Apply WoE transformation adding group id for each variable in dataset.
		For modelling purposes only WoE is required.

		Parameters
		----------
		data : pandas.DataFrame
			Dataset used for performing optimal grouping or new dataset to
			apply transformations on variables matching the original dataset.

		only_woe : boolean (default=True)
			If True, the transformed dataset only includes the WoE
			transformation. Otherwise, WoE and group id, i.e., split id, is
			added.

		add_target : boolean (default=False)
			Whether to add the target column used while performing the optimal
			grouping process.

		Returns
		-------
		self : object

		Warning
		-------
		Currently not implemented.				
		"""
		raise NotImplementedError

	def _run(self, file_format):
		"""Run optimal grouping + checks."""
		time_run = time.perf_counter()

		# run OptBin algorithm for each variable in dataset
		if self.verbose:
			print("running OptimalGrouping...")

		for id, name in enumerate(self._column_names):
			if self.verbose:
				print("\nvariable {}: {}".format(id, name))

			if file_format == "csv":
				self._variables_information.append(self._run_variable(
					self._read_column_csv(name), name))
			elif file_format == "parquet":
				self._variables_information.append(self._run_variable(
					self._read_column_parquet(name), name))

		# statistics
		self._statistics()

		self._time_run = time.perf_counter() - time_run

		# update flag
		self._is_run = True

		return self

	def _run_variable(self, x, name):
		"""
		Run Optimal binning optimizer (OptBin) for a given variable using
		default settings.

		Note: for numerical and ordinal variable types OptBin uses RTree as
		a prebinning algorithm and CTree otherwise.
		"""

		# variable type
		dtype = check_dtype(name, x.dtype, self.variables_nominal,
			self.verbose)

		if self.verbose:
			print("variable dtype:", dtype)

		# instantiate and configure OptBin
		if self._target_dtype == "binary":
			if dtype in ("categorical", "nominal"):
				optbin_solver = OptBin(name=name,
					dtype=dtype,
					prebinning_algorithm="ctree",
					ctree_min_criterion=0.95,
					ctree_max_candidates=64,
					ctree_dynamic_split_method="entropy+k-tile",
					monotonicity_force=self.monotonicity_force,
					special_values=self.special_values,
					special_handler_policy=self.special_handler_policy,
					special_woe_policy=self.special_woe_policy,
					missing_woe_policy=self.missing_woe_policy,
					verbose=self.verbose)
			else:
				optbin_solver = OptBin(name=name,
					dtype=dtype,
					monotonicity_force=self.monotonicity_force,
					special_values=self.special_values,
					special_handler_policy=self.special_handler_policy,
					special_woe_policy=self.special_woe_policy,
					missing_woe_policy=self.missing_woe_policy,
					verbose=self.verbose)
		elif self._target_dtype == "numerical":
			if dtype in ("categorical", "nominal"):
				optbin_solver = OptBinContinuous(name=name,
					dtype=dtype,
					prebinning_algorithm="ctree",
					ctree_min_criterion=0.95,
					ctree_max_candidates=64,
					ctree_dynamic_split_method="entropy+k-tile",
					monotonicity_force=self.monotonicity_force,
					special_values=self.special_values,
					special_handler_policy=self.special_handler_policy,
					special_woe_policy=self.special_woe_policy,
					missing_woe_policy=self.missing_woe_policy,
					verbose=self.verbose)
			else:
				optbin_solver = OptBinContinuous(name=name,
					dtype=dtype,
					monotonicity_force=self.monotonicity_force,
					special_values=self.special_values,
					special_handler_policy=self.special_handler_policy,
					special_woe_policy=self.special_woe_policy,
					missing_woe_policy=self.missing_woe_policy,
					verbose=self.verbose)

		optbin_solver.fit(x, self._target)

		# timing
		if dtype is "numerical":
			self._n_vars_numerical += 1
			self._time_run_numerical += optbin_solver._time_total
		elif dtype is "ordinal":
			self._n_vars_ordinal += 1
			self._time_run_ordinal += optbin_solver._time_total
		elif dtype is "nominal":
			self._n_vars_nominal += 1
			self._time_run_nominal += optbin_solver._time_total
		elif dtype is "categorical":
			self._n_vars_categorical += 1
			self._time_run_categorical += optbin_solver._time_total

		return optbin_solver

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

		# select target data. save target to avoid querying several times.
		self._target = self._read_column_csv(self.target)

		# check target type
		self._target_type = check_target_dtype(self._target)

		if self.verbose:
			print("target type is {}.".format(self._target_type))

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
		self._n_vars = len(self._column_names)

		if not self.target in self._column_names:
			raise ValueError("target column {} not available in dataframe."
				.format(self.target))

		# select target data. save target to avoid querying several times.
		self._target = self._read_column_parquet(self.target)

		# check target type
		self._target_dtype = check_target_dtype(self._target)

		if self.verbose:
			print("target type is {}.".format(self._target_type))

		# exclude target from the list of variables
		self._column_names.remove(self.target)

		self._n_samples= len(self._target)

	def _read_column_csv(self, column):
		"""Read an specific column from CSV file."""
		return pd.read_csv(filepath_or_buffer=self.path_csv, sep=self.sep_csv,
			engine='c', encoding='latin-1', low_memory=False, memory_map=True,
			usecols=[column]).values

	def _read_column_parquet(self, column):
		"""Read an specific column from Parquet file."""
		return pq.read_table(source=self.path_parquet,
			columns=[column]).to_pandas().iloc[:, 0].values

	def _read_header_csv(self):
		"""Read CSV header."""
		return list(pd.read_csv(filepath_or_buffer=self.path_csv,
			sep=self.sep_csv, engine='c', encoding='latin-1', low_memory=False,
			memory_map=True, header=None, nrows=1).values[0])
