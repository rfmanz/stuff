"""
Mathematical optimization framework for solving the optimal binning problem
with continuous target.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import os
import re
import sys
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull
from scipy.stats import mannwhitneyu
from scipy.stats import median_test
from scipy.stats import ttest_ind_from_stats
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeRegressor

from ..._thirdparty.mipcl.mipshell import *
from ...core.base import GRMlabBase
from ...core.exceptions import NotFittedException
from .binning import plot
from .binning import table
from .ctree import CTree
from .rtree_categorical import RTreeCategorical
from .util import process_data


def weighted_median(data, weights):
    """
    Compute weighted median.

	data : list or numpy.ndarray

	weights : list or numpy.ndarray
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()

    if data.size == 1:
    	return data

    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median


def interpolation_median(medians, records):
	"""
	Linear interpolation to estimate median of grouped data.

	Parameters
	----------
	medians : list or numpy.ndarray
		median of each bucket.

	records : list or numpy.ndarray
		records of each bucket.
	"""
	rcumsum = np.cumsum(records)
	midpoint = rcumsum[-1] / 2
	j = np.searchsorted(rcumsum, midpoint)
	width = medians[j] - medians[j - 1]
	lower = medians[j-1]
	slope = (midpoint - rcumsum[j - 1]) / records[j] * width
	return lower + slope


def non_parametric_test(x, y, splits, special_values=None, test="median"):
	x, y, _ = process_data(x, y, None, special_values)

	max_pvalue = 0

	if isinstance(splits[0], (list, np.ndarray)):
		for i in range(len(splits) - 1):
			y1 = y[pd.Series(x).isin(splits[i]).values]
			y2 = y[pd.Series(x).isin(splits[i+1]).values]

			if test == "mann_whitney":
				pvalue = mannwhitneyu(y1, y2)[1]
			elif test == "median":
				pvalue = median_test(y1, y2)[1]

			if pvalue > max_pvalue:
				max_pvalue = pvalue
	else:
		bands = np.insert(splits, 0, np.min(x))
		bands = np.append(bands, np.max(x))

		for i in range(len(bands) - 2):
			if i == 0:
				y1 = y[(x <= bands[i+1])]
			else:
				y1 = y[(x > bands[i]) & (x <= bands[i+1])]

			y2 = y[(x > bands[i+1]) & (x <= bands[i+2])]

			if test == "mann_whitney":
				pvalue = mannwhitneyu(y1, y2)[1]
			elif test == "median":
				pvalue = median_test(y1, y2)[1]

			if pvalue > max_pvalue:
				max_pvalue = pvalue

	return max_pvalue


class MIPCLSolver(Problem):
	"""
	MIPCL representation of the Optimal Binning optimization problem. A set 
	of useful structures for building the problem is included.
	"""	
	@staticmethod
	def build(records, sums, medians, sds, metric, median_method, max_pvalue,
		pvalue_constraint_method):
		"""
		Generate auxiliary data: aggregated PD and IV matrices and indicator
		lists for constructing constraints.
		"""
		n = len(sums)
		nodes = n * (n + 1) // 2
		inner_nodes = n * (n - 1) // 2

		U = []
		R = []
		S = []
	
		for i in range(1, n+1):
			for j in range(i):
				# mean or median grouped metric
				if metric == "mean":
					u = sum(sums[k] for k in range(j, i)) / sum(records[k]
						for k in range(j, i))
				elif metric == "median":
					md = [medians[k] for k in range(j, i)]

					if median_method == "weighted_mean":
						wd = [records[k] for k in range(j, i)]
						u = np.average(md, weights=wd)
					elif median_method == "mean":
						u = np.mean(md)
					elif median_method == "median":
						u = np.median(md)
					elif median_method == "weighted_median":
						wd = [records[k] for k in range(j, i)]
						u = weighted_median(md, weights=wd)
					elif median_method == "interpolation":
						wd = [records[k] for k in range(j, i)]
						u = interpolation_median(md, wd)
				U.append(u)

				# data for max p-value constraints
				if max_pvalue is not None:
					# number of records
					wd = [records[k] for k in range(j, i)]
					R.append(sum(wd))

					if pvalue_constraint_method == "frequency":
						# standard deviation estimate assuming normal distribution
						sx2f = sum(sums[k] ** 2 / records[k] for k in range(j, i))
						sxf2 = sum(sums[k] for k in range(j, i)) ** 2 / R[-1]
						ssx = sx2f - sxf2
						ss = np.sqrt(ssx / (R[-1] - 1))
					elif pvalue_constraint_method == "weighted_mean":
						sd = [sds[k] for k in range(j, i)]
						ss = np.average(sd, weights=wd)
					S.append(ss)

		# diagonal
		diag = [(i-1)*(i+2)//2 for i in range(1, n+1)]

		# lower triangular matrix
		low_tri = [i for i in range(nodes) if i not in diag]

		# rows
		rows = [[0]]
		r = list(range(nodes))
		for i in range(1,n):
			rows.append(r[diag[i-1]+1:diag[i]+1])

		# cols
		cols = []
		for j in range(n):
			z = j*(j+3)//2
			k = 0
			col = []
			for i in range(1, n-j+1):
				col.append(z+k)
				k += j+i
			cols.append(col)

		# p-value indexes
		pvalue_indexes = []
		if max_pvalue is not None:
			for i, row in enumerate(rows[:-1]):
				for r in row:
					for j in cols[i+1]:
						if ttest_ind_from_stats(U[r], S[r], R[r], U[j], S[j],
							R[j], equal_var=False)[1] > max_pvalue:
							pvalue_indexes.append((r, j))

		return U, diag, low_tri, rows, cols, sum(records), pvalue_indexes

	def model(self, records, sums, medians, sds, M, M2, sense, minbs, maxbs,
		mincs, maxcs, fixed, reduce_bucket_size_diff, t, metric, median_method,
		max_pvalue, pvalue_constraint_method):
		"""
		MILP formulation in MIPshell for the optimal binning problem. 
		Several formulations are available, including 3 different objective
		functions. The code is not self-explanatory, and it is not aimed to be.
		"""

		# build problem data and indicator lists
		U, diag, low_tri, rows, cols, total, pvalues = self.build(records, sums,
			medians, sds, metric, median_method, max_pvalue,
			pvalue_constraint_method)
		self.diag = diag

		# parameters
		n = len(records)
		nvs = n * (n + 1) // 2

		# auxiliary variables
		self.xx = xx = [1] * n

		# decision variables
		# ==================
		self.x = x = VarVector([nvs], "x", BIN)
		self.tm = tm = VarVector([n], "tm", REAL, lb=0)
		self.tp = tp = VarVector([n], "tp", REAL, lb=0)

		# auxiliary variables for inequality constraints
		if maxcs is not None and mincs is not None:
			self.d = d = Var("d", INT, lb=0, ub=maxcs - mincs)

		# auxiliary variables for reducing pmax-pmin
		if reduce_bucket_size_diff:
			self.pmin = pmin = Var("pmin")
			self.pmax = pmax = Var("pmax")

		# objective function
		# ======================================================================
		if reduce_bucket_size_diff:
			minimize(sum_(tp[i] + tm[i] for i in range(n)) + (pmax - pmin))
		else:
			minimize(sum_(tp[i] + tm[i] for i in range(n)))

		# constraints
		# ======================================================================

		# objective function linearization absolute value
		for i, row in enumerate(rows):
			tp[i] - tm[i] - U[row[-1]] * (1 - x[row[-1]]) + sum_(
				(U[j] - U[j+1]) * x[j] for j in row[:-1]) == 0

		# all sum of columns = 1
		for col in cols:
			sum_(x[i] for i in col) == 1

		# flow continuity
		for i in low_tri:
			x[i+1] - x[i] >= 0

		# monotonicity constraints
		if sense is "ascending":
			for i in range(1, n):
				row = rows[i]
				for p_row in rows[:i]:
					M + (U[row[-1]]-M) * x[row[-1]] + sum_(
						(U[j] - U[j+1]) * x[j] for j in row[:-1]
					) >= U[p_row[-1]] * x[p_row[-1]] + sum_(
						(U[j] - U[j+1]) * x[j] for j in p_row[:-1]
						) + M2 * (x[row[-1]] + x[p_row[-1]] -2)

		elif sense is "descending":
			for i in range(1, n):
				row = rows[i]
				for p_row in rows[:i]:
					U[row[-1]] * x[row[-1]] + sum_(
						(U[j] - U[j+1]) * x[j] for j in row[:-1]) + M2 * (
						x[row[-1]] + x[p_row[-1]] - 2
						) <= M + (U[p_row[-1]] - M) * x[p_row[-1]] + sum_(
						(U[j] - U[j+1]) * x[j] for j in p_row[:-1])

		elif sense is "peak":
			for i in range(1, t):
				row = rows[i]
				for p_row in rows[:i]:
					M + (U[row[-1]]-M) * x[row[-1]] + sum_(
						(U[j] - U[j+1]) * x[j] for j in row[:-1]
					) >= U[p_row[-1]] * x[p_row[-1]] + sum_(
						(U[j] - U[j+1]) * x[j] for j in p_row[:-1]
						) + M2 * (x[row[-1]] + x[p_row[-1]] -2)

			for i in range(t, n):
				row = rows[i]
				for p_row in rows[t:i]:
					U[row[-1]] * x[row[-1]] + sum_(
						(U[j] - U[j+1]) * x[j] for j in row[:-1]) + M2 * (
						x[row[-1]] + x[p_row[-1]] - 2
						) <= M + (U[p_row[-1]] - M) * x[p_row[-1]] + sum_(
						(U[j] - U[j+1]) * x[j] for j in p_row[:-1])

		elif sense is "valley":
			for i in range(1, t):
				row = rows[i]
				for p_row in rows[:i]:
					U[row[-1]] * x[row[-1]] + sum_(
						(U[j] - U[j+1]) * x[j] for j in row[:-1]) + M2 * (
						x[row[-1]] + x[p_row[-1]] - 2
						) <= M + (U[p_row[-1]] - M) * x[p_row[-1]] + sum_(
						(U[j] - U[j+1]) * x[j] for j in p_row[:-1])

			for i in range(t, n):
				row = rows[i]
				for p_row in rows[t:i]:
					M + (U[row[-1]]-M) * x[row[-1]] + sum_(
						(U[j] - U[j+1]) * x[j] for j in row[:-1]
					) >= U[p_row[-1]] * x[p_row[-1]] + sum_(
						(U[j] - U[j+1]) * x[j] for j in p_row[:-1]
					) + M2 * (x[row[-1]] + x[p_row[-1]] -2)

		# min/max buckets
		if maxcs is not None and mincs is not None:
			d + sum_(x[i] for i in diag) == maxcs
		elif maxcs is not None:
			sum_(x[i] for i in diag) <= maxcs
		elif mincs is not None:
			sum_(x[i] for i in diag) >= mincs

		# reduce diff between largest and smallest bucket size
		if reduce_bucket_size_diff:
			for i in range(n):
				sum_(records[k] * x[j] for k, j in enumerate(rows[i])
					) <= pmax
				sum_(records[k] * x[j] for k, j in enumerate(rows[i])
					) >= pmin - total / n * (1-x[diag[i]])			

		# max bucket size
		if maxbs < total:
			for i in range(n):
				sum_(records[k] * x[j] for k, j in enumerate(rows[i])
					) <= maxbs * x[diag[i]]

		for i in range(n):
			sum_(records[k] * x[j] for k, j in enumerate(rows[i])
				) >= minbs * x[diag[i]]

		# max p-value
		for i, j in pvalues:
			x[i] + x[j] <= 1

		# preprocessing
		# ======================================================================

		# fixed buckets
		if fixed:
			for i in fixed:
				x[diag[i]] == 1

	def solution(self):
		return [int(self.x[i].val) for i in self.diag]

	def infeasible_buckets(self):
		return len(self.xx) - sum(self.xx)

	def run(self, timeLimit, silent=False):
		# run optimizer
		self.optimize(timeLimit=timeLimit, silent=silent)
		# remove mp
		self.mp = None


class OptBinContinuous(GRMlabBase):
	"""
	OptBin algorithm to perform optimal feature binning with continuous
	target.

	OptBin solves a mixed-integer linear programming (IP/MILP) optimization
	problem via a rigorous approach.

	Parameters
	----------
	name : str
		The variable name.

	dtype : str (default="numerical")
		The variable data type. Four dtypes supported: "categorical", "nominal",
		"numerical" and "ordinal".

	metric : str (default="mean")
		The metric that must satisfy monotonicity constraints. Two metrics
		supported: "mean" and "median".

	median_method : str (default="weighted_mean")
		The method to estimate the median of the optimal buckets. Four
		methods: "mean" (mean of medians), "median" (median of medians),
		"weighted mean" (weighted mean of medians), "weighted_median"
		(weighted median of medians) and "interpolation". The weighted methods
		use the number of records of prebinning buckets as weights.

	prebinning_algorithm : str (default="rtree")
		The prebinning algorithm. Algorithm supported are: "ctree" and "rtree".
		Option "ctree" uses ``grmlab.data_processing.feature_binning.CTree``
		algorithm, whereas "rtree" uses CART algorithm implementations
		``sklearn.tree.DecisionTreeRegressor`` and
		``grmlab.data_processing.feature_binning.RTreeCategorical``.

	rtree_max_leaf_nodes : int (default=20)
		The maximum number of leaf nodes.

	ctree_min_criterion: float (default=0.95)
		The value of the test statistic or 1 - (alpha or significance level)
		that must be exceeded in order to add a split.

	ctree_max_candidates : int (default=64)
		The maximum number of split points to perform an exhaustive search.

	ctree_dynamic_split_method : str (default="k-tile")
		The method to generate dynamic split points. Supported methods are
		“gaussian” for the Gaussian approximation, “k-tile” for the quantile
		approach and “entropy+k-tile” for a heuristic using class entropy. The
		"entropy+k-tile" method is only applicable when target is binary,
		otherwise, method "k-tile" is used instead.

	prebinning_others_group : boolean (default=True)
		Whether to create an extra group with those values (categories) do not
		sufficiently representative. This option is available for dtypes
		"categorical" and "nominal".

	prebinning_others_threshold : float (default=0.01)
		Merge categories which percentage of cases is below threshold to create
		an extra group. This option is available for dtypes "categorical" and
		"nominal".

	min_buckets : int or None (default=None)
		The minimum number of optimal buckets. If None then unlimited number of
		buckets.

	max_buckets : int or None (default=None)
		The maximum number of optimal buckets. If None then unlimited number of
		buckets.

	min_bucket_size : float (default=0.05)
		The minimum number of records per bucket. Percentage of total records.

	max_bucket_size : float (default=1.0)
		The maximum number of records per bucket. Percentage of total records.

	monotonicity_sense : str (default="auto")
		The event rate monotonicity sense. Supported options are: "auto",
		"ascending", "descending", "peak" and "valley".
		Option "auto" estimates the monotonicity sense minimizing MSE.

	monotonicity_force : boolean (default=False)
		Force ascending or descending monotonicity when
		``monotonicity_sense="auto"``.

	reduce_bucket_size_diff : boolean (default=False)
		Try to reduce standard deviation among size of optimal buckets. This
		aims to produce solutions with more homogeneous bucket sizes, thus
		avoiding cases where only a few buckets are dominant.

	pvalue_method : str (default="welch_t")
		The statistical test to determine whether two consecutive buckets are
		significantly different. Three method are supported: "welch_t" uses
		Welch's t-test,  "mann_whitney" uses Mann-Whitney rank test and "median"
		using Mood's median test. 

	max_pvalue : float or None (default=None):
		The maximum allowed p-value between consecutive buckets.

	pvalue_constraint_method : str (default="weighted_mean")
		The method to estimate the standard deviation required to compute the
		p-value between consecutive optimal buckets, using the Welch's t-test.
		Two methods: "weighted_mean" and "frequency".

	user_splits : list or None (default=None)
		List of prebinning split points provided by a user.

	user_idx_forced_splits : list or None (default=None)
		Indexes of split points to be fixed. The optimal solution must include
		those splits.

	special_values : list or None (default=None)
		List of special values to be considered.

	special_handler_policy : str (default="join")
		Method to handle special values. Options are "join", "separate" and
		"binning". Option "join" creates an extra bucket containing all special
		values. Option "separate" creates an extra bucket for each special
		value. Option "binning" performs feature binning of special values using
		``grmlab.data_processing.feature_binning.CTree`` in order to split
		special values if these are significantly different.

	verbose: int or boolean (default=False)
		Controls verbosity of output.

	See also
	--------
	CTree, OptBin

	Notes
	-----
	When ``metric=="median"`` the median of the optimal buckets is estimated.
	Note that there is no absolute guarantee to obtain a result satisfying
	monotonicity constraints, although it might work reasonably well for
	most cases if ``median_method=="weighted_mean"``. Additionally, for
	``metric=="median"`` is convenient to use a non-parametric test
	(Mann-Whitney or median test).
	"""
	def __init__(self, name="", dtype="numerical", metric="mean",
		median_method="weighted_mean", prebinning_algorithm="rtree",
		rtree_max_leaf_nodes=20, ctree_min_criterion=0.95, ctree_max_candidates=64,
		ctree_dynamic_split_method="k-tile", prebinning_others_group=True,
		prebinning_others_threshold=0.01, min_buckets=None, max_buckets=None,
		min_bucket_size=0.05, max_bucket_size=1.0, monotonicity_sense="auto",
		monotonicity_force=False, reduce_bucket_size_diff=False, 
		pvalue_method="welch_t", max_pvalue=0.05,
		pvalue_constraint_method="weighted_mean", user_splits=None,
		user_idx_forced_splits=None, special_values=None,
		special_handler_policy="join", special_woe_policy="empirical",
		missing_woe_policy="empirical", verbose=False):

		# main input data
		self.name = name
		self.dtype = dtype

		# monotonicity metric
		self.metric = metric
		self.median_method = median_method

		# pre-binning options (activated)
		self._is_prebinning = True
		self.prebinning_algorithm = prebinning_algorithm

		# rtree parameters
		self.rtree_max_leaf_nodes =  rtree_max_leaf_nodes

		# ctree options
		self.ctree_min_criterion = ctree_min_criterion
		self.ctree_max_candidates = ctree_max_candidates
		self.ctree_dynamic_split_method = ctree_dynamic_split_method
		self.prebinning_others_group = prebinning_others_group
		self.prebinning_others_threshold = prebinning_others_threshold

		# general optbin parameters
		self.monotonicity_user = monotonicity_sense
		self.monotonicity_sense = monotonicity_sense
		self.monotonicity_force = monotonicity_force
		self.min_buckets = min_buckets
		self.max_buckets = max_buckets
		self.min_bucket_size = min_bucket_size
		self.max_bucket_size = max_bucket_size
		self.pvalue_method = pvalue_method
		self.max_pvalue = max_pvalue
		self.pvalue_constraint_method = pvalue_constraint_method
		self.reduce_bucket_size_diff = reduce_bucket_size_diff

		# user-defined splits
		self.user_splits_provided = False
		self.user_splits = [] if user_splits is None else user_splits
		if user_idx_forced_splits is None:
			self.user_idx_forced_splits = []
		else:
			self.user_idx_forced_splits = user_idx_forced_splits

		# special values options
		self.special_values = [] if special_values is None else special_values
		self.special_handler_policy = special_handler_policy
		self.special_woe_policy = special_woe_policy

		# missing values options
		self.missing_woe_policy = missing_woe_policy

		# others
		self.verbose = verbose

		# main dataframe/array characteristics
		self._n_samples= None

		# peak/valley extra parameter
		self._trend_change = None

		# MIPCL solver options
		self._time_limit = 60

		# problem status
		self._is_solution = False
		self._is_solution_optimal = False
		self._is_infeasible = False
		self._is_unbounded = False

		# problem statistics
		self._mipcl_problem = None
		self._mipcl_msg = None
		self._mipcl_obj = None
		self._n_prebuckets = None
		self._n_optimal_buckets = None
		self._infeasible_buckets = None
		self._nvs = None
		self._ncs = None
		self._nnz = None
		self._nvs_preprocessing = None
		self._ncs_preprocessing = None
		self._nnz_preprocessing = None
		self._nvs_removed = None
		self._ncs_removed = None
		self._nnz_removed = None
		self._cuts_generated = None
		self._cuts_used = None
		self._branch_and_cut_nodes = None
		self._iters_time = []
		self._iters_obj = []
		self._iters_cuts = []

		# optimal-binning results
		self._binning_table_optimal = None
		self._max_pvalue = None
		self._largest_bucket_perc = None
		self._smallest_bucket_perc = None
		self._largest_bucket_size = None
		self._smallest_bucket_size = None
		self._diff_largest_smallest_bucket_size = None
		self._std_bucket_size = None
		self._group_special = None
		self._group_missing = None
		self._group_others = None

		# pre-binning results
		self._prebinning_trend_changes = None

		# timing statistics
		self._time_total = None
		self._time_prebinning = None
		self._time_problem_data = None
		self._time_solver = None
		self._time_problem_generation = None
		self._time_optimizer_preprocessing = None
		self._time_optimizer = None
		self._time_post_analysis = None

		# cutpoints / splits variables
		self._splits_prebinning = []
		self._splits_optimal = []
		self._splits_specials = []
		self._splits_others = []

		# flags
		self._is_fitted = False

	def fit(self, x, y, sample_weight=None, check_input=True):
		"""
		Build optimal binning from the training set (x, y).

		Parameters
		----------
		x : array-like, shape = [n_samples]
			The training input samples.

		y : array-like, shape = [n_samples]
			The target values.

		check_input : boolean (default=True)
			Option to perform several input checking.

		Returns
		-------
		self : object
		"""		
		if check_input:
			if not isinstance(x, np.ndarray):
				raise TypeError("x must be a numpy array.")

			if not isinstance(y, np.ndarray):
				raise TypeError("y must be a numpy array.")

			if not x.size:
				raise ValueError("x cannot be empty.")

			if not y.size:
				raise ValueError("y cannot be empty.")

			if len(x) != len(y):
				raise ValueError("x and y must have the same length.")

		# variable dtype
		if self.dtype not in ("numerical", "ordinal", "nominal", "categorical"):
			raise ValueError("dtype not supported.")

		# metric
		if self.metric not in ("mean", "median"):
			raise ValueError("metric not supported.")

		if self.median_method not in ("mean", "median", "weighted_mean",
			"weighted_median", "interpolation"):
			raise ValueError("median method not supported.")

		# pre-binning algorithm
		if self.prebinning_algorithm not in ("rtree", "ctree"):
			raise ValueError("prebinning_algorithm not supported.")

		# pre-binning or user-defined splits
		if len(self.user_splits):
			if self._is_prebinning and self.verbose:
				print("pre-binning algorithm is disable.")

			self.user_splits_provided = True
			self._is_prebinning = False

		# monotonicity
		if self.monotonicity_sense not in ("auto", "ascending", "descending", 
			"peak", "valley"):
			raise ValueError("monotonicity sense {} not supported. "
				   "Available options: auto, ascending, descending"
				   " valley and peak.".format(
				   	self.monotonicity_sense))

		# max/min buckets
		if self.min_buckets is not None and self.max_buckets is not None:
			if self.min_buckets > self.max_buckets:
				raise ValueError("min_buckets must be <= max_buckets.")

		if self.min_buckets is not None and self.min_buckets < 0:
			raise ValueError("min_buckets must be > 0.")

		if self.max_buckets is not None and self.max_buckets < 0:
			raise ValueError("max_buckets must be > 0.")

		# max/min bucket size
		if self.min_bucket_size > self.max_bucket_size:
			raise ValueError("min_bucket_size must be <= max_bucket_size.")

		if self.min_bucket_size < 0 or self.min_bucket_size > 1.0:
			raise ValueError("min_bucket_size must be in (0, 1).")

		if self.max_bucket_size < 0 or self.max_bucket_size > 1.0:
			raise ValueError("max_bucket_size must be in (0, 1).")

		# p-value method
		if self.pvalue_method not in ("welch_t", "mann_whitney", "median"):
			raise ValueError("pvalue_method not supported.")

		# max p-value
		if self.max_pvalue is not None:
			if self.max_pvalue < 0.0 or self.max_pvalue > 1.0:
				raise ValueError("max_pvalue must be a float in [0, 1].")

		# p-value constraint method
		if self.pvalue_constraint_method not in ("weighted_mean", "frequency"):
			raise ValueError("pvalue_constraint_method not supported.")

		# ctree_min_criterion
		if self.ctree_min_criterion < 0.5 or self.ctree_min_criterion > 1.0:
			raise ValueError("ctree_min_criterion must be a float in [0.5, 1.0).")

		# ctree_max_candidates
		if self.ctree_max_candidates < 2:
			raise ValueError("ctree_max_candidates must be >= 2.")

		# ctree_dynamic_split_method
		if self.ctree_dynamic_split_method not in ("gaussian", "k-tile", 
			"entropy+k-tile"):
			raise ValueError("ctree_dynamic_split_method not supported.")

		# special values
		if not isinstance(self.special_values, (list, np.ndarray)):
			raise TypeError("special_values must be a list of numpy array.")

		# special_handler_policy
		if self.special_handler_policy not in ("join", "separate", "binning"):
			raise ValueError("special_handler_policy option not supported.")

		# special_woe_policy
		if self.special_woe_policy not in ("empirical", "worst", "zero"):
			raise ValueError("special_woe_policy option not supported.")

		# missing_woe_policy
		if self.missing_woe_policy not in ("empirical", "worst", "zero"):
			raise ValueError("missing_woe_policy option not supported.")

		# fit optbin
		return self._fit(x, y)			

	def _fit(self, x, y):
		"""Start solving the Optimal Binning problem."""
		time_init = time.perf_counter()
		
		x = x.copy()
		y = y.copy()
		self._n_samples= len(x)

		# if user-defined splits are provided do not perform pre-binning, 
		# otherwise run pre-binning solving classification decision tree
		if self.user_splits_provided:
			self._splits_prebinning = self.user_splits
			self._group_others = 0
			self._time_prebinning = 0
		else:
			self._is_prebinning = True
			self._prebinning(x, y)

		# solve IP optimization problem using MIPCL solver
		self._optimize(x, y)

		# parse solver output to compute problem statistics
		self._statistics(x, y)

		self._time_total = time.perf_counter() - time_init

		# update flag
		self._is_fitted = True

		return self

	def stats(self):
		"""OptBin solver and timing statistics."""
		if not self._is_fitted:
			raise NotFittedException(self)
		
		self._stats_report()

	def transform(self, x, metric=None):
		"""
		Apply transformation to array x.

		Parameters
		----------
		x : array-like, shape = [n_samples]
			The training input samples.

		metric : str or None (default=None)
			The metric to apply transformation. Two metrics supported: "mean"
			and "median". If None, the metric selected to satisfy monotonicity
			constraints is used.
		"""
		if not self._is_fitted:
			raise NotFittedException(self)

		if not isinstance(x, np.ndarray):
			raise TypeError("x must be a numpy array.")

		if metric is not None:
			if metric not in ("mean", "median"):
				raise ValueError("metric {} not supported.".format(metric))

		n = len(x)

		# indexes with special values
		idx_special = np.isin(x, self.special_values)

		# optimal splits
		n_splits = len(self.splits_optimal)

		# metric from binning table
		if metric is not None:
			if metric == "mean":
				woe = self._binning_table_optimal.Mean.values[:-1]
			elif metric == "median":
				woe = self._binning_table_optimal.Median.values[:-1]
		elif self.metric == "mean":
			woe = self._binning_table_optimal.Mean.values[:-1]
		elif self.metric == "median":
			woe = self._binning_table_optimal.Median.values[:-1]

		# initialize woe and groups arrays
		t_woe = np.zeros(n)

		if not n_splits:
			t_woe = np.ones(n) * woe[0]
			idx_nan = pd.isnull(x)
			n_splits = 1

		elif self.dtype in ("categorical", "nominal"):
			# categorical and nominal variables return groups as numpy.ndarray
			for _idx, split in enumerate(self.splits_optimal):
				# mask = np.isin(x, split)
				mask = pd.Series(x).isin(split).values
				t_woe[mask] = woe[_idx]
			# indexes with NaN in x
			idx_nan = pd.isnull(x)

		else:
			# numerical and ordinal variables return extra group (> last split)
			splits = self.splits_optimal[::-1]
			mask = (x > splits[-1])
			t_woe[mask] = woe[n_splits]
			for _idx, split in enumerate(splits):
				mask = (x <= split)
				t_woe[mask] = woe[n_splits - (_idx + 1)]
			# indexes with NaN in x
			idx_nan = pd.isnull(x)
			# account for > group
			n_splits += 1

		# special values
		if self._splits_specials:
			for _idx, split in enumerate(self._splits_specials):
				if isinstance(split, np.ndarray):
					mask = np.isin(x, split)
				else:
					mask = (x == split)
				t_woe[mask] = woe[n_splits]
				n_splits += 1
		else:
			t_woe[idx_special] = woe[n_splits]
			n_splits += 1

		# missing values
		t_woe[idx_nan] = woe[n_splits]

		return t_woe

	def binning_table(self):
		"""
		Return binning table with optimal buckets / split points.

		Returns
		-------
		binning_table : pandas.DataFrame
		"""
		if not self._is_fitted:
			raise NotFittedException(self)
		
		return self._binning_table_optimal

	def plot_binning_table(self, metric=None):
		"""
		Plot showing how the set of pre-buckets is merged in order to satisfy
		all constraints.

		Parameters
		----------
		metric : str or None (default=None)
			The metric to plot. Two metrics supported: "mean" and "median".
			If None, the metric selected to satisfy monotonicity constraints
			is plotted.
		"""
		if not self._is_fitted:
			raise NotFittedException(self)

		if metric is not None:
			if metric not in ("mean", "median"):
				raise ValueError("metric {} not supported.".format(metric))

			lb_feature = "Mean" if metric == "mean" else "Median"
		else:
			lb_feature = "Mean" if self.metric == "mean" else "Median"

		return plot(self._binning_table_optimal, self._splits_optimal,
			plot_type="default", others_group=self._group_others,
			lb_feature=lb_feature)

	def plot_optimizer_progress(self):
		"""
		Plot showing how the solver proceeds to the optimal solution. It also
		includes the number of generated cuts at each iteration.
		"""
		if not self._is_fitted:
			raise NotFittedException(self)

		return self._plot_solver_progress(self._iters_time, 
			self._iters_obj, self._iters_cuts)

	def _prebinning(self, x, y):
		"""
		Run pre-binning phase to generate set of pre-buckets to be merged in
		the optimization phase.
		"""
		time_init = time.perf_counter()

		# minimum number of samples required to be a leaf nodes: is computed
		# from min_bucket_size
		min_samples_leaf = int(self.min_bucket_size * self._n_samples)

		# remove NaN and special values from data
		x, y, _ = process_data(x, y, None, self.special_values)

		y = y / np.linalg.norm(y)

		if self.prebinning_algorithm == "rtree":
			if self.dtype in ("nominal", "categorical"):
				rtree = RTreeCategorical(min_samples_leaf=min_samples_leaf,
					max_leaf_nodes=self.rtree_max_leaf_nodes, presort=True)

				rtree.fit(x, y)

				if rtree.others_group:
					self._group_others = 1
					self._splits_prebinning = rtree.splits[:-1]
					self._splits_others = rtree.splits[-1]

					if self.verbose:
						print("pre-binning: group others was generated " 
							"including {} distinct values.".format(
								len(self._splits_others)))
				else:
					self._group_others = 0
					self._splits_prebinning = rtree.splits
			else:
				# build and fit decision tree classifier (CART)
				rtree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf,
					max_leaf_nodes=self.rtree_max_leaf_nodes, presort=True)

				rtree.fit(x.reshape(-1, 1), y)

				# retrieve splits
				splits = np.unique(rtree.tree_.threshold)
				self._splits_prebinning = splits[splits != _tree.TREE_UNDEFINED]

				# others group
				self._group_others = 0

		else:
			# build and fit GRMlab CTree
			ctree = CTree(dtype=self.dtype,
				min_samples_leaf=min_samples_leaf, 
				min_criterion=self.ctree_min_criterion,
				max_candidates=self.ctree_max_candidates,
				dynamic_split_method=self.ctree_dynamic_split_method,
				others_group=self.prebinning_others_group,
				others_threshold=self.prebinning_others_threshold,
				verbose=self.verbose)

			ctree.fit(x, y)

			# retrieve splits
			if self.dtype in ("nominal", "categorical"):
				if ctree.others_group:
					self._group_others = 1
					self._splits_prebinning = ctree.splits[:-1]
					self._splits_others = ctree.splits[-1]

					if self.verbose:
						print("pre-binning: group others was generated " 
							"including {} distinct values.".format(
								len(self._splits_others)))
				else:
					self._group_others = 0
					self._splits_prebinning = ctree.splits
			else:
				self._group_others = 0
				self._splits_prebinning = ctree.splits

		self._time_prebinning = time.perf_counter() - time_init

	def _pre_optimize_data(self, x, y):
		"""Compute data to pass to the solver."""
		time_init = time.perf_counter()

		records, sums, means, medians, sds = self._check_prebinning(x, y)

		# extra parameters for the optimization problem
		total = sum(records)
		min_bucket_size = int(self.min_bucket_size * total)
		max_bucket_size = int(self.max_bucket_size * total)

		if self.metric == "mean":
			metric_values = means
		else:
			metric_values = medians

		# big-M
		bigM = np.absolute(metric_values).max()
		if all(np.array(metric_values) > 0):
			bigM2 = 0
		elif all(np.array(metric_values) < 0):
			bigM2 = bigM
			bigM = 0
		else:	
			bigM2 = bigM

		# monotonicity sense: compute decision variables and apply decision
		# tree classifier to determine the sense maximizing IV.
		if self._n_prebuckets == 0:
			if self.verbose:
				print("pre-binning analysis: it is not possible to create any",
					"significant split. A unique group is returned.")
				print("pre-binning analysis: all splits were removed.")
			self.monotonicity_sense = "undefined"

		elif self._n_prebuckets == 1:
			if self.verbose:
				print("pre-binning analysis: it is not possible to create any", 
					"significant split. A unique group is returned.")
				print("pre-binning analysis: monotonicity is set to ascending.")
			self.monotonicity_sense = "ascending"

		elif self.monotonicity_sense == "auto":
			if self.dtype in ("nominal", "categorical"):
				self.monotonicity_sense = "descending"
			else:
				params = self._monotonicity_parameters(metric_values, records)
				self.monotonicity_sense = self._monotonicity_decision(*params)

			if self.verbose:
				print("pre-binning analysis: monotonicity decision engine",
					"detects {} sense.".format(self.monotonicity_sense))

		elif self.monotonicity_sense in ("peak", "valley"):
			self._compute_trend_change(metric_values)

		# compute trend changes in pre-binning solution
		self._prebinning_trend_changes = self._trend_changes(means)	

		self._time_problem_data = time.perf_counter() - time_init

		return records, sums, medians, sds, min_bucket_size, max_bucket_size, bigM, bigM2

	def _check_prebinning(self, x, y):
		"""
		"""
		meant, _ = table(x, y, self._splits_prebinning, None,
			self.special_handler_policy, self.special_values,
			self.special_woe_policy, self.missing_woe_policy)

		# extract records, sum and mean by split
		if self.dtype in ("nominal", "categorical"):
			n_buckets = len(self._splits_prebinning)
		else:	
			n_buckets = len(self._splits_prebinning) + 1

		self._n_prebuckets = n_buckets

		records = list(meant.iloc[:, 1].values[:n_buckets])
		sums = list(meant.iloc[:, 3].values[:n_buckets])
		means = list(meant.iloc[:, 4].values[:n_buckets])
		medians = list(meant.iloc[:, 5].values[:n_buckets])
		sds = list(meant.iloc[:, 6].values[:n_buckets])

		return records, sums, means, medians, sds

	def _optimize(self, x, y):
		"""Build MIPCL model and solve."""
		time_init = time.perf_counter()

		# compute optimization problem data
		[records, sums, medians, sds, min_bucket_size, max_bucket_size, bigM,
			bigM2] = self._pre_optimize_data(x, y)

		# no need to solve optimization problem, no buckets.
		if self._n_prebuckets in [0, 1]:
			if self.verbose:
				print("optimization: prebuckets <= 1, no optimization",
					"is required.")

			self._is_solution = True
			self._is_solution_optimal = True
			self._splits_optimal = self._splits_prebinning

			# add group others if generated
			if self._group_others:
				self._splits_optimal = list(self._splits_optimal) + [
					self._splits_others]

			self._infeasible_buckets = 0
		else:
			# initialize optimization problem
			self._mipcl_problem = MIPCLSolver("OptBin")

			if self.monotonicity_sense in ("peak", "valley") and self.verbose:
				print("pre-binning analysis: trend_change position =", 
						self._trend_change)

			self._mipcl_problem.model(records, sums, medians, sds, bigM, bigM2,
				self.monotonicity_sense, min_bucket_size, max_bucket_size,
				self.min_buckets, self.max_buckets,
				self.user_idx_forced_splits, self.reduce_bucket_size_diff,
				self._trend_change, self.metric, self.median_method,
				self.max_pvalue, self.pvalue_constraint_method)

			# run solver and catch C++ std output for post-analysis
			self._run_optimizer()

			# retrieve problem solution and preprocessing information
			solution = self._mipcl_problem.solution()
			self._infeasible_buckets = self._mipcl_problem.infeasible_buckets()

			try:
				if self.dtype in ("numerical", "ordinal"):
					for idx, v in enumerate(solution[:-1]):
						if v:
							self._splits_optimal.append(
								self._splits_prebinning[idx])
				else:
					add_bucket = []
					for idx, v in enumerate(solution):
						if v:
							if add_bucket:
								new_bucket = sum([add_bucket, 
									list(self._splits_prebinning[idx])], [])
								self._splits_optimal.append(np.array(new_bucket))
								add_bucket = []
							else:
								self._splits_optimal.append(
									self._splits_prebinning[idx])
						else:
							add_bucket += list(self._splits_prebinning[idx])
			except Exception as e:
				# TODO: change method to capture this type of errors.
				# user should be aware of data types, otherwise run block
				# using optimalgrouping.
				print("optimize: check variable type and/or pre-binning algorithm.")
				raise

			# add group others if generated
			if self._group_others:
				self._splits_optimal += [self._splits_others]

			# post-solve information
			self._is_solution = self._mipcl_problem.is_solution
			self._is_solution_optimal = self._mipcl_problem.is_solutionOptimal
			self._is_infeasible = self._mipcl_problem.is_infeasible
			self._is_unbounded = self._mipcl_problem.is_unbounded

			# is solution is optimal get value otherwise return NaN, this will
			# be output for infeasible and unbounded problems.
			if self._is_solution_optimal:
				self._mipcl_obj = self._mipcl_problem.getObjVal()
			else:
				self._mipcl_obj = np.NaN

		self._time_solver = time.perf_counter() - time_init

	def _run_optimizer(self):
		"""
		Capture C++ std output from the MIPCL solver. Use mutable objects 
		through threads. Jupyter Notebook replaces std I/O by their own
		custom implementations. The following codes circumvents this issue,
		however, last output line is not captured and appears first in 
		subsequent runs, this issue need to be taken into account when parsing
		the log file.
		"""

		# prepare std output

		# file descriptor unique id (UNIX / Windows) is sys.stdout.fileno(), 
		# however, as aforementioned this would not work in Jupyter. By default,
		# fileno = 1 ==> standard output.
		stdout_fileno = 1
		stdout_save = os.dup(stdout_fileno)
		stdout_pipe_read, stdout_pipe_write = os.pipe()

		# copy current stdout
		os.dup2(stdout_pipe_write, stdout_fileno)
		os.close(stdout_pipe_write)

		# prepare list to collect solver messages
		msgs = []

		# trigger thread
		t = threading.Thread(target=self._catpure_solver_output, 
			args=(stdout_pipe_read, msgs, ))

		t.start()

		# run optimizer
		self._mipcl_problem.run(timeLimit=self._time_limit, silent=False)
		# close stdout and collect thread
		os.close(stdout_fileno)
		t.join()

		# clean up the pipe and restore original stdout
		os.close(stdout_pipe_read)
		os.dup2(stdout_save, stdout_fileno)
		os.close(stdout_save)

		# construct output message (small fix is required)
		self._mipcl_msg = ''.join(msgs)

	def _statistics(self, x, y):
		"""Collect OptBin and MIPCL statistics."""
		time_init = time.perf_counter()

		if self._n_prebuckets >= 2:
			stats = self._parser_solver_output()

			# problem statistics
			self._ncs = stats[0]
			self._nvs = stats[1]
			self._nnz = stats[2]
		else:
			self._ncs = self._n_prebuckets
			self._nvs = self._n_prebuckets
			self._nnz = self._n_prebuckets

		if self._is_solution_optimal and self._n_prebuckets >= 2:

			self._ncs_preprocessing = stats[3]
			self._nvs_preprocessing = stats[4]
			self._nnz_preprocessing = stats[5]
			self._ncs_removed = self._ncs_preprocessing - self._ncs
			self._nvs_removed = self._nvs_preprocessing - self._nvs
			self._nnz_removed = self._nnz_preprocessing - self._nnz

			# objective value, cuts and time per iteration
			self._iters_time = stats[6]
			self._iters_obj = stats[7]
			self._iters_cuts = stats[8]

			# cuts information
			self._cuts_generated = stats[9]
			self._cuts_used = stats[10]
			self._branch_and_cut_nodes = stats[13]

			# timing
			self._time_optimizer_preprocessing = stats[11]
			self._time_optimizer = stats[12]
			self._time_problem_generation = self._time_solver
			self._time_problem_generation -= self._time_optimizer
			self._time_problem_generation -= self._time_problem_data

			# add final iteration
			self._iters_time.append(self._time_optimizer)
			self._iters_obj.append(self._mipcl_obj)
			self._iters_cuts.append(np.NaN)

		else:  # infeasible / unbounded case
			self._mipcl_obj = np.NaN

			self._ncs_preprocessing = np.NaN
			self._nvs_preprocessing = np.NaN
			self._nnz_preprocessing = np.NaN
			self._ncs_removed = np.NaN
			self._nvs_removed = np.NaN
			self._nnz_removed = np.NaN

			self._iters_time = np.NaN
			self._iters_obj = np.NaN
			self._iters_cuts = np.NaN

			self._cuts_generated = 0
			self._cuts_used = 0
			self._branch_and_cut_nodes = 0

			self._time_optimizer_preprocessing = 0
			self._time_optimizer = 0
			self._time_problem_generation = self._time_solver
			self._time_problem_generation -= self._time_problem_data

		# optimal-binning results
		if self.dtype in ("numerical", "ordinal"):
			self._n_optimal_buckets = len(self._splits_optimal) + 1
		else:
			self._n_optimal_buckets = len(self._splits_optimal)

		self._binning_table_optimal, self._splits_specials = table(x, y, 
			self._splits_optimal, None, self.special_handler_policy, 
			self.special_values, self.special_woe_policy,
			self.missing_woe_policy)

		# p-value computation
		if self._is_solution_optimal and len(self._splits_optimal):
			tb = self._binning_table_optimal.iloc[:,
				[1, 4, 6]].values[:self._n_optimal_buckets]

			if len(tb) <= 1:
				self._max_pvalue = 0
			elif self.pvalue_method == "welch_t":
				self._max_pvalue = max(ttest_ind_from_stats(tb[i, 1],
					tb[i, 2], tb[i, 0], tb[i+1, 1], tb[i+1, 2], tb[i+1, 0], 
					equal_var=False)[1] for i in range(len(tb)-1))
			elif self.pvalue_method in ("mann_whitney", "median"):
				self._max_pvalue = non_parametric_test(x, y, self._splits_optimal,
					self.special_values, self.pvalue_method)

		# buckets size information
		if self._is_solution_optimal:
			if self._n_prebuckets:
				records = self._binning_table_optimal.iloc[:, 
					1].values[:self._n_optimal_buckets]
			else:
				records = self._binning_table_optimal.iloc[:, 
					1].values[:self._n_optimal_buckets+1]

			self._largest_bucket_size = np.max(records)
			self._smallest_bucket_size = np.min(records)
			self._largest_bucket_perc = self._largest_bucket_size / self._n_samples
			self._smallest_bucket_perc = self._smallest_bucket_size / self._n_samples
			self._diff_largest_smallest_bucket_size = self._largest_bucket_size
			self._diff_largest_smallest_bucket_size -= self._smallest_bucket_size
			self._std_bucket_size = np.std(records)

		else:
			self._largest_bucket_size = self._smallest_bucket_size = 1.0
			self._diff_largest_smallest_bucket_size = 0.0
			self._std_bucket_size = 0.0

		# missing / specials and others buckets
		records = self._binning_table_optimal.iloc[:, 1].values[:-1]

		if self._n_prebuckets:
			records_specials = records[self._n_optimal_buckets:-1]
		else:
			records_specials = records[self._n_optimal_buckets+1:-1]

		records_missing = records[-1]
		self._group_special = sum(1 for w in records_specials if w)
		self._group_missing = 1 if records_missing else 0

		self._time_post_analysis = time.perf_counter() - time_init

	def _parser_solver_output(self):
		"""Parser for MIPCL output."""
		lines = self._mipcl_msg.split("\n")

		is_start_pre = False
		is_after_pre = False
		is_optimizing = False
		is_generating_cuts = False
		is_cut_statistics = False
		
		ncs_original = None
		nvs_original = None
		nnz_original = None
		ncs_after = None
		nvs_after = None
		nnz_after = None
		
		iters_time = []
		iters_obj = []
		iters_cuts = []
		
		cuts_generated = None
		cuts_used = None
		
		preprocessing_time = None
		solution_time = None
		branch_and_cut_nodes = None
		
		regex_basic = re.compile("(\d+)")
		regex_decimal = re.compile("(0|([1-9]\d*))(\.\d+)?")

		for line in lines:
			if "Start preprocessing" in line: 
				if not is_start_pre:
					data = regex_basic.findall(line)
					ncs_original = int(data[0])
					nvs_original = int(data[1])
					nnz_original = int(data[4])
					is_start_pre = True
				else: continue
				
			if "After preprocessing" in line:
				if not is_after_pre:
					data = regex_basic.findall(line)
					ncs_after = int(data[0])
					nvs_after = int(data[1])
					nnz_after = int(data[4])
					is_after_pre = True
				else: continue
				
			if "Preprocessing Time" in line:
				preprocessing_time = float(regex_decimal.search(line).group(0))
			
			elif "Optimizing..." in line:
				is_optimizing = True
				
			elif is_optimizing and not is_cut_statistics:
				data = [f.group(0) for f in regex_decimal.finditer(line)]
				if len(data) == 4:
					if float(data[1]) == int(float(data[1])):
						iters_time.append(float(data[0]))
						iters_obj.append(float(data[3]))
						iters_cuts.append(np.NaN)
					else:
						iters_time.append(float(data[0]))
						iters_obj.append(float(data[1]))
						iters_cuts.append(int(data[3]))
						
			if "Cut statistics" in line:
				is_cut_statistics = True
				is_optimizing = False
				
			elif is_cut_statistics:
				if "total" in line:
					cuts_generated, cuts_used = list(
						map(int, regex_basic.findall(line)))
					is_cut_statistics = False
					
			elif "Solution time" in line:
				solution_time = float(regex_decimal.search(line).group(0))
			
			elif "Branch-and-Cut nodes" in line:
				branch_and_cut_nodes = int(regex_basic.search(line).group(0))
		
		return [ncs_original, nvs_original, nnz_original, ncs_after, nvs_after, 
				nnz_after, iters_time, iters_obj, iters_cuts, cuts_generated, 
				cuts_used, preprocessing_time, solution_time, 
				branch_and_cut_nodes]		

	def _stats_report(self):
		"""
		Report OptBin statistics. NOTE: output should be improved in future
		releases, current approach is hard to maintain.
		"""
		prebinning = "yes" if self._is_prebinning else "no"
		minbs = "not set" if self.min_buckets is None else self.min_buckets
		maxbs = "not set" if self.max_buckets is None else self.max_buckets 

		if self.monotonicity_sense is "ascending":
			sense = "asc"
		elif self.monotonicity_sense is "descending":
			sense = "desc"
		elif self.monotonicity_sense is "peak":
			sense = "peak"
		elif self.monotonicity_sense is "valley":
			sense = "valley"
		elif self.monotonicity_sense is "undefined":
			sense = "undefined"

		if self.monotonicity_user is "ascending":
			user_sense = "asc"
		elif self.monotonicity_user is "descending":
			user_sense = "desc"
		elif self.monotonicity_user is "peak":
			user_sense = "peak"
		elif self.monotonicity_user is "valley":
			user_sense = "valley"			
		else:
			user_sense = "auto"

		spec = "yes" if self.special_values else "no"
		prebuckets = "yes" if self.user_splits_provided else "no"
		indexforced = "yes" if self.user_idx_forced_splits else "no"
		reduce_bucket_diff = "yes" if self.reduce_bucket_size_diff else "no"
		max_pvalue = self.max_pvalue if self.max_pvalue is not None else "no"

		# pre-binning algorithm options
		if self.prebinning_algorithm is "ctree":
			ct_vartype = self.dtype
			ct_mincrit = round(self.ctree_min_criterion, 8)
			ct_maxcand = self.ctree_max_candidates
			ct_dynamic = self.ctree_dynamic_split_method
		else:
			ct_vartype = "not set"
			ct_mincrit = "not set"
			ct_maxcand = "not set"
			ct_dynamic = "not set"

		if not self._is_prebinning:
			pre_algo = "not set"
		else:
			pre_algo = self.prebinning_algorithm

		# cuts are not always generated
		if not self._cuts_generated:
			cut_g = 0
			cut_u = 0
			cut_p = 0
		else:
			cut_g = self._cuts_generated
			cut_u = self._cuts_used
			cut_p = cut_u / cut_g

		# % time spent in preprocessing w.r.t optimizer
		if self._time_optimizer:
			rel_pre = self._time_optimizer_preprocessing / self._time_optimizer
		else:
			rel_pre = 0

		# buckets reduction
		buckets_reduction = self._n_optimal_buckets - self._n_prebuckets

		# optimization problem status
		status = "stopped"
		if self._is_solution_optimal:
			status = "optimal"
		elif self._is_infeasible:
			status = "infeasible"
		elif self._is_unbounded:
			status = "unbounded"

		report = (
		"\033[94m================================================================================\033[0m\n"
		"\033[1m\033[94m             GRMlab OptBin Continuous 0.1: Feature binning optimizer            \033[0m\n"
		"\033[94m================================================================================\033[0m\n"
		"\n"
		" \033[1mMain options                              Extra options\033[0m\n"
		"   pre-binning              {:>4}             special values               {:>3}\n"
		"   pre-binning max nodes    {:>4}             special handler policy{:>10}\n"
		"   monotonicity sense    {:>7}             special WoE policy    {:>10}\n"
		"   min buckets           {:>7}             missing WoE policy    {:>10}\n"
		"   max buckets           {:>7}\n"
		"   min bucket size         {:>4.3f}           \033[1mUser pre-binning options\033[0m\n"
		"   max bucket size         {:>4.3f}             pre-buckets                  {:>3}\n"
		"   reduce bucket size diff  {:>4}             indexes forced               {:>3}\n"
		"   max p-value              {:>4}\n"
		"   p-value method  {:>13}\n"
		"   p-value const.  {:>13}\n"
		"\n"
		" \033[1mPre-binning algorithmic options\033[0m\n"
		"   algorithm             {:>7}\n"
		"   ctree options\n"
		"     variable type  {:>12}\n"
		"     min criterion      {:>8}\n"
		"     max candidates      {:>7}\n"
		"     DSM          {:>14}\n"
		"   \033[94m--------------------------------------------------------------------------\033[0m\n"
		"\n"
		" \033[1mProblem statistics                        Optimizer statistics\033[0m\n"
		"   original problem                          status                {:>10}\n"
		"     variables          {:>6}               objective           {:>12.5E}\n"
		"     constraints        {:>6}\n"
		"     nonzeros           {:>6}\n"
		"   after preprocessing                       cutting planes\n"
		"     variables          {:>6} ({:>7})       cuts generated     {:>4}\n"
		"     constraints        {:>6} ({:>7})       cuts used          {:>4} ({:>4.0%})\n"
		"     nonzeros           {:>6} ({:>7})     branch-and-cut nodes {:>4}\n"
		"\n"
		" \033[1mTiming statistics                         Optimizer options\033[0m\n"
		"   total                {:>6.3f}               root LP algorithm   {:>10}\n"
		"     prebinning         {:>6.3f} ({:>4.0%})        time limit            {:>10}\n" 
		"     model data         {:>6.3f} ({:>4.0%})        MIP gap                 {:>7.6f}\n"
		"     model generation   {:>6.3f} ({:>4.0%})                                        \n"
		"     optimizer          {:>6.3f} ({:>4.0%})                                        \n"
		"       preprocessing      {:>6.3f} ({:>4.0%})                                      \n"
		"     post-analysis      {:>6.3f} ({:>4.0%})                                        \n"
		"   \033[94m--------------------------------------------------------------------------\033[0m\n"
		"\n"
		" \033[1mPre-binning statistics                    Optimal-binning statistics\033[0m\n"
		"   buckets              {:>6}               buckets               {:>3} ({:>4})\n"
		"   trend changes        {:>6}               monotonicity           {:>9}\n"
		"                                             p-value                  {:>6.5f}\n"
		"\n"
		"                                             largest bucket     {:>6} ({:>4.0%})\n"
		"                                             smallest bucket    {:>6} ({:>4.0%})\n"
		"                                             std bucket size       {:>10.2f}\n"
		"   \033[94m--------------------------------------------------------------------------\033[0m\n"
		).format(prebinning, spec, self.rtree_max_leaf_nodes,
		self.special_handler_policy, user_sense, self.special_woe_policy,
		minbs, self.missing_woe_policy, maxbs, self.min_bucket_size,
		self.max_bucket_size, prebuckets, reduce_bucket_diff, indexforced,
		max_pvalue, self.pvalue_method, self.pvalue_constraint_method,
		pre_algo, ct_vartype, ct_mincrit, ct_maxcand, ct_dynamic, status,
		self._nvs, self._mipcl_obj, self._ncs, self._nnz,
		self._nvs_preprocessing, self._nvs_removed, cut_g,
		self._ncs_preprocessing, self._ncs_removed, cut_u, cut_p,
		self._nnz_preprocessing, self._nnz_removed, self._branch_and_cut_nodes,
		self._time_total, "Dual-Simplex", self._time_prebinning,
		self._time_prebinning/self._time_total, self._time_limit,
		self._time_problem_data, self._time_problem_data/self._time_total, 0, 
		self._time_problem_generation, self._time_problem_generation/self._time_total,
		self._time_optimizer, self._time_optimizer/self._time_total,
		self._time_optimizer_preprocessing, rel_pre, self._time_post_analysis, 
		self._time_post_analysis/self._time_total, self._n_prebuckets,
		self._n_optimal_buckets, buckets_reduction,
		self._prebinning_trend_changes, sense, self._max_pvalue,
		self._largest_bucket_size, self._largest_bucket_perc,
		self._smallest_bucket_size, self._smallest_bucket_perc,
		self._std_bucket_size)

		print(report)

	def _compute_trend_change(self, metric):
		"""Compute trend change split for valley/peak non-rigorous approach."""
		if self.monotonicity_sense is "peak":
			self._trend_change = np.argmax(metric)
		elif self.monotonicity_sense is "valley":
			self._trend_change = np.argmin(metric)

	def _monotonicity_parameters(self, metric, records):
		"""
		Compute parameters needed by the decision tree to determine the correct
		monotonicity sense when "auto" mode is active.
		"""

		# curve-fitting and quadratict improvement
		x = np.arange(len(metric), dtype=float)
		metric = np.asarray(metric, dtype=float)
		linear = np.polyfit(x, metric, deg=1, full=True)
		quadratic = np.polyfit(x, metric, deg=2, full=True)

		linear_residual = linear[1]
		quadratic_residual = quadratic[1]

		diff_fitting = abs(linear_residual - quadratic_residual)
		quadratic_improvement = diff_fitting / linear_residual

		linear_coef = linear[0][0]
		quadratic_coef = quadratic[0][0]
		linear_sense = "descending" if linear_coef < 0 else "ascending"
		quadratic_sense = "peak" if quadratic_coef < 0 else "valley"

		# triangle area: abs(0.5 * det[A]))
		x0 = metric[0]; y0 = 0
		xn = metric[-1]; yn = metric.size
		xmin = metric.min(); ymin = metric.argmin()
		xmax = metric.max(); ymax = metric.argmax()

		Amin = np.array([[x0, xmin, xn], [y0, ymin, yn], [1, 1, 1]])
		Amax = np.array([[x0, xmax, xn], [y0, ymax, yn], [1, 1, 1]])
		area_min = 0.5 * abs(np.linalg.det(Amin))
		area_max = 0.5 * abs(np.linalg.det(Amax))

		# dominant triangle
		area = max(area_min, area_max)
		area_total = (xmax-xmin) * yn
		area_ratio = area / area_total

		# percentage at both sides of trend checkpoint
		if area == area_min: # area = are_min => valley
			left = sum(records[:ymin])
			right = sum(records[ymin+1:])

			left_elements = records[:ymin]
			right_elements = records[ymin+1:]

			left_mean = np.mean(left_elements) if len(left_elements) else 0
			right_mean = np.mean(right_elements) if len(right_elements) else 0
		else: # area = are_max => peak
			left = sum(records[:ymax])
			right = sum(records[ymax+1:])

			left_elements = records[:ymax]
			right_elements = records[ymax+1:]

			left_mean = np.mean(left_elements) if len(left_elements) else 0
			right_mean = np.mean(right_elements) if len(right_elements) else 0

		# mean_left/mean and mean_right/mean
		mean_total = np.mean(records)
		mean_left_mean = left_mean / mean_total
		mean_right_mean = right_mean / mean_total

		# compute convex hull
		len_pds = len(metric)
		points = np.zeros((len_pds, 2))
		points[:, 0] = np.arange(len_pds)
		points[:, 1] = metric

		total_area = (xmax - xmin) * yn

		if len_pds > 2:
			try:
				hull = ConvexHull(points)
				convexhull_area = hull.volume / total_area
			except:
				convexhull_area = 0
		else:
			convexhull_area = 0

		# trend change peak/valley
		if quadratic_sense is "peak":
			self._trend_change = ymax
		elif quadratic_sense is "valley":
			self._trend_change = ymin

		return [linear_sense, quadratic_sense, quadratic_improvement, area,
			left, right, mean_left_mean, mean_right_mean, convexhull_area]

	def _monotonicity_decision(self, linear_sense, quadratic_sense,
		quadratic_improvement, triangle_area, left, right, mean_left_mean,
		mean_right_mean, convexhull_area):
		"""
		Decision tree to decide which monotonicity sense is most sensible to
		maximize IV.

		monotonicity:
			+ ascending / descending = 1
			+ convex / concave or peak / valley = 0

		Note: as the number of cases increases the algorithm shall improve
		its predictive power.
		"""
		monotonicity_sense = 0

		if right <= 0.076:
			if convexhull_area <= 0.473:
				monotonicity_sense = 1
			else:  # if convexhull_area > 0.47283774614334106
				monotonicity_sense = 0
		else:  # if right > 0.07573056221008301
			if mean_left_mean <= 0.778:
				if quadratic_improvement <= 0.058:
					monotonicity_sense = 1
				else:  # if quadratic_improvement > 0.05759747326374054
					monotonicity_sense = 0
			else:  # if mean_left_mean > 0.7778470516204834
				monotonicity_sense = 0

		# analyze monotonicity and print result
		if monotonicity_sense == 1 or self.monotonicity_force:
			return linear_sense  # asc / desc
		else:
			# peak/valley and self._iv_prebinning >= 0.05
			return quadratic_sense

	@staticmethod
	def _catpure_solver_output(stdout_pipe_read, msgs):
		"""Capture solver messages from the pipe."""
		nbytes = 1024
		encoding = "utf-8"
		while True:
			data = os.read(stdout_pipe_read, nbytes)
			if not data:
				break
			msgs.append(data.decode(encoding))

	@staticmethod
	def _trend_changes(x):
		"""
		Detect the number of trend changes from the PD curve computed after
		performing pre-binning.
		"""
		n = len(x)
		n1 = n-1

		n_asc = n_des = 0
		peaks = valleys = 0

		for i in range(1, n1):
			if x[i] < x[i-1] and x[i] < x[i+1]:
				valleys += 1
			if x[i] > x[i-1] and x[i] > x[i+1]:
				peaks += 1
		changes = peaks + valleys

		return changes

	@staticmethod
	def _plot_solver_progress(time, obj, cuts):
		"""
		Plot MIPCL progress in terms of objective function and cut generation.
		"""
		fig, ax = plt.subplots(1,1)
		ax.plot(time, obj, '-xb', label="objective")
		ax2 = ax.twinx()
		ax2.plot(time, cuts, '^c', label="generated cuts")

		# label
		ax.set_ylabel("Objective value")
		ax2.set_ylabel("Cuts")
		ax.set_xlabel("time (s) - after preprocessing")

		# legend
		lines, labels = ax.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		lines += lines2
		labels += labels2
		ax2.legend(lines, labels)

		plt.show()

	@property
	def splits_optimal(self):
		"""
		OptBin splits points.

		Returns
		-------
		splits : numpy.ndarray
		"""
		return np.asarray(self._splits_optimal)

	@property
	def splits_prebinning(self):
		"""
		Prebinning splits points.

		Returns
		-------
		splits : numpy.ndarray
		"""
		return np.asarray(self._splits_prebinning)
