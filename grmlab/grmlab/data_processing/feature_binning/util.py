"""
Feature binning module utils

BBVA - GRM Team Copyright 2018.
"""
import numpy as np
import pandas as pd


def _rounding(x):
	"""Returns rounding to show at least 4 significant digits."""
	rounding = 0
	if len(x) > 1:
		rounding = int(np.log10(np.min(np.abs(x[x!=0]))))
	elif x[0] != 0:
		rounding = int(np.log10(abs(x[0])))
	else:
		return 0

	if rounding <= 0:
		# return -rounding + 1
		return -rounding + 2
	else:
		# return max(4 - rounding, 0)
		return 2


def _binary(x):
	"""Check whether an array contains only binary data."""
	u = np.unique(x) # fastest for general purposes
	binary_01 = (len(u) == 2 and u[0] == 0 and u[1] == 1)
	binary_0 = all(x==0)
	binary_1 = all(x==1)
	return binary_01 or binary_0 or binary_1


def process_data(x, y, sample_weight=None, spec_values=None):
	"""
	Returns clean data after removing NaNs (missing values) and special values.
	"""

	# nan records
	if isinstance(x.dtype, object):
		nan_idx = (pd.isnull(x) | np.isnan(y))
	else:
		nan_idx = (np.isnan(x) | np.isnan(y))
	# special values
	if spec_values:
		spe_idx = (np.isin(x, spec_values))
		# numeric records
		x = x[~nan_idx & ~spe_idx]
		y = y[~nan_idx & ~spe_idx]
		if sample_weight is not None:
			sample_weight = sample_weight[~nan_idx & ~spe_idx]
	else:
		x = x[~nan_idx]
		y = y[~nan_idx]
		if sample_weight is not None:
			sample_weight = sample_weight[~nan_idx]

	return x, y, sample_weight


def _separated_data(x, y, sample_weight=None, spec_values=None):
	"""Returns clean, missing and special data with the corresponding target."""

	if sample_weight is None:
		sample_weight = np.ones(x.shape[0])
	
	# nan records
	if isinstance(x.dtype, object):
		nan_idx = (pd.isnull(x) | np.isnan(y))
	else:
		nan_idx = (np.isnan(x) | np.isnan(y))
	x_nan = x[nan_idx]
	y_nan = y[nan_idx]
	weight_nan = sample_weight[nan_idx]

	# special values
	spe_idx = (np.isin(x, spec_values))
	x_spec = x[spe_idx]
	y_spec = y[spe_idx]
	weight_spec = sample_weight[spe_idx]

	# numeric records
	x = x[~nan_idx & ~spe_idx]
	y = y[~nan_idx & ~spe_idx]
	sample_weight = sample_weight[~nan_idx & ~spe_idx]

	return x, y, sample_weight, x_nan, y_nan, weight_nan, x_spec, y_spec, weight_spec


def categorical_others_group(x, y, sample_weight=None, threshold=0.01, verbose=False):
	"""
	Identify non representative groups and join them as other category.
	TODO: IMPROVE METHODOLOGY.
	"""
	n = len(x)
	# values, counts = np.unique(x, return_counts=True)
	uniq_c = pd.value_counts(x)
	values, counts = uniq_c.index.values, uniq_c.values

	if len(values) > 2:
		# sufficient distinct values to generate splits
		# min_threshold = np.sort(counts)[::-1][2] / n
		min_threshold = counts[2] / n
		new_threshold = min(threshold, min_threshold)
		if new_threshold != threshold and verbose:
			print("CTree: others_threshold was updated to {}".format(
				new_threshold))

		# filter values to be joined and remove from list of values to build
		# tree.
		# https://stackoverflow.com/questions/50779617/pandas-pd-series-isin-performance-with-set-versus-array
		others = values[counts < int(new_threshold * n)]
		idx_others = pd.Series(x).isin(others).values

		if sample_weight is not None:
			sample_weight = sample_weight[~idx_others]

		return x[~idx_others], y[~idx_others], sample_weight, others
	else:
		return x, y, sample_weight, []


def _check_input(df, value_name, target_name):
	"""Check whether df is a dataframe and data and target are in dataframe."""
	if not isinstance(df, pd.DataFrame):
		raise ValueError("df must be a pandas DataFrame.")

	if not isinstance(value_name, str):
		raise ValueError("value_name must be a string.")

	if not value_name in list(df.columns.values):
		raise ValueError("value_name not in df.")

	if not target_name in list(df.columns.values):
		raise ValueError("target_name not in df.")

	if not isinstance(splits, (list, np.ndarray)):
		raise ValueError("splits must be a list or numpy array.")


def _check_input_table(df):
	"""Check whether dataframe comes from binning table."""
	if not isinstance(df, pd.DataFrame):
		raise ValueError("df must be a pandas DataFrame.")

	header = list(df.columns.values)
	HEADER_WOE = ["WoE", "Event count", "Default rate"]
	HEADER_DEFAULT = ["WoE", "Event count", "Default rate"]

	if (not all(h in header for h in HEADER_WOE) and 
		not all(h in header for h in HEADER_DEFAULT)):
		raise ValueError("df must be the result of type table.")


def _check_input_tree(values, target):
	"""Check whether data for partitioning tree algorithms is valid."""
	if not isinstance(values, np.ndarray):
		raise TypeError("values must be a numpy array.")

	if not isinstance(target, np.ndarray):
		raise TypeError("target must be a numpy array")

	# check if empty or different dimensions
	if values.size == 0:
		raise ValueError("values cannot be empty.")

	if target.size == 0:
		raise ValueError("target cannot be empty.")

	if len(values) != len(target):
		raise ValueError("values and target must have the same length.")
