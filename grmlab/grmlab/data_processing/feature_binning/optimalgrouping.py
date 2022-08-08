"""
Optimal Grouping automatic process.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2018.

import time

from copy import deepcopy

import numpy as np
import pandas as pd

from ...core.base import GRMlabBase
from ...core.dtypes import check_dtype
from ...core.dtypes import check_target_dtype
from ...core.dtypes import is_binary
from ...core.exceptions import NotRunException
from ...reporting.util import reporting_output_format
from .optbin import OptBin
from .optbin_continuous import OptBinContinuous


class OptimalGrouping(GRMlabBase):
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

    continuous_metric : str (default="mean")
        The metric that must satisfy monotonicity constraints. Two metrics
        supported: "mean" and "median". This option applies when target is
        continuous.

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

    optbin_options : dict or None (default=None):
        Dictionary with options and comments to pass to a particular optbin
        instance.

    verbose : int or boolean (default=False)
        Controls verbosity of output.

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
        monotonicity_force=False, continuous_metric="mean",
        special_handler_policy="join", special_woe_policy="empirical",
        missing_woe_policy="empirical", optbin_options=None, verbose=False):

        # main input
        self.target = target
        self._target_dtype = None
        self._target = []

        # nominal variables
        if variables_nominal is None:
            self.variables_nominal = []
        else:
            self.variables_nominal = variables_nominal

        self.monotonicity_force = monotonicity_force

        self.continuous_metric = continuous_metric

        # special values options
        self.special_values = [] if special_values is None else special_values
        self.special_handler_policy = special_handler_policy
        self.special_woe_policy = special_woe_policy

        # missing values options
        self.missing_woe_policy = missing_woe_policy

        # optbin options
        self.optbin_options = optbin_options

        # others
        self.verbose = verbose

        # main dataframe characteristics
        self._n_samples = None
        self._n_vars = None

        # descriptive statistics
        self._n_vars_numerical = 0
        self._n_vars_ordinal = 0
        self._n_vars_nominal = 0
        self._n_vars_categorical = 0

        # summary
        self._summary_table = None

        # iv analysis
        self._iv_min = None
        self._iv_max = None
        self._iv_mean = None
        self._iv_median = None
        self._iv_small = None
        self._iv_small_perc = None
        self._iv_large = None
        self._iv_large_perc = None

        # monotonicity analysis
        self._n_vars_ascending = None
        self._n_vars_descending = None
        self._n_vars_peak = None
        self._n_vars_valley = None
        self._n_vars_undefined = None
        self._pd_asc_perc = None
        self._pd_desc_perc = None
        self._pd_peak_perc = None
        self._pd_valley_perc = None
        self._pd_undefined_perc = None

        # p-value analysis
        self._pvalue_min = None
        self._pvalue_max = None
        self._pvalue_small = None
        self._pvalue_small_perc = None
        self._pvalue_large = None
        self._pvalue_large_perc = None

        # groups analysis
        self._groups_max = None
        self._groups_min = None
        self._groups_0 = None
        self._groups_1 = None
        self._groups_special = None
        self._groups_missing = None
        self._groups_others = None
        self._groups_0_perc = None
        self._groups_1_perc = None
        self._groups_missing_perc = None
        self._groups_special_perc = None
        self._groups_others_perc = None

        # timing statistics
        self._time_run = None
        self._time_run_numerical = 0
        self._time_run_ordinal = 0
        self._time_run_nominal = 0
        self._time_run_categorical = 0

        # auxiliary variables (flags)
        self._is_run = False

        # auxiliary information
        self._column_names = []

        # list of variables with optbin information
        self._variables_information = []

        self._dict_optbin_options = {}

    def results(self, format="dataframe"):
        """
        Return information and flags for all variables binned using OptBin.

        Parameters
        ----------
        format : str, "dataframe" or "json" (default="dataframe")
            If "dataframe" return pandas.DataFrame. Otherwise, return serialized
            json.
        """
        if not self._is_run:
            raise NotRunException(self)

        summary = self._summary_table

        # order by IV descending
        if self._target_dtype == "binary":
            summary = summary.sort_values(by=["IV", "variable"], 
                ascending=False).reset_index(drop=True)
        else:
            summary = summary.sort_values(by=["variable"], 
                ascending=False).reset_index(drop=True)

        return reporting_output_format(summary, format)     

    def run(self, data, sample_weight=None):
        """
        Run optimal automatic grouping.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataset to perform optimal grouping.

        sample_weight : array-like, shape = [n_samples] (default=None)
            Individual weights for each sample. It is implemented only for
            binary target.

        Returns
        -------
        self : object
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas.DataFrame.")

        # check sample_weight
        if sample_weight is not None:
            if not isinstance(sample_weight, (list, np.ndarray)):
                raise TypeError("sample_weight must be a list or a numpy"
                                "array.")
            else:
                sample_weight = np.array(sample_weight)
                if sample_weight.shape[0] != data.shape[0]:
                    raise ValueError("sample_weight len and data rows number"
                                     "do not match.")

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

        # run optimal grouping
        return self._run(data, sample_weight)

    def stats(self):
        """OptimalGrouping statistics."""
        if not self._is_run:
            raise NotRunException(self)

        self._stats_report()

    def transform(self, data, only_metric=True, add_target=False):
        """
        Apply WoE transformation adding group id for each variable in dataset.
        For modelling purposes only WoE is required.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataset used for performing optimal grouping or new dataset to
            apply transformations on variables matching the original dataset.

        only_metric : boolean (default=True)
            If True, the transformed dataset only includes the metric
            transformation. Otherwise, metric and group id, i.e., split id, is
            added. The metric is WoE and mean for binary and numerical target,
            respectively.

        add_target : boolean (default=False)
            Whether to add the target column used while performing the optimal
            grouping process.

        Notes
        -----
        All operations are performed **in-place**, therefore one must create a 
        copy of the provided dataframe in case is needed for posterior usage.
        """
        if not self._is_run:
            raise NotRunException(self)
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas dataframe.")

        n = len(data)

        for variable in self._variables_information:
            # variable information
            varname = variable.name
            dtype = variable.dtype

            if self.verbose:
                print("variable: ", varname)

            # data
            x = data[varname]

            if dtype is "categorical":
                # numpy bug with categorical data => force casting
                idx_special = np.isin(x, self.special_values)
                idx_nan = pd.isnull(x)
                idx_clean = (~idx_nan) & (~idx_special)
                x[idx_clean] = x[idx_clean].astype(str)

            # indexes with special values
            idx_special = np.isin(x, self.special_values)

            # header name
            grp_name = "GRP_{}".format(varname)

            # optimal splits
            n_splits = len(variable.splits_optimal)

            if self._target_dtype == "binary":
                # woe from binning table
                woe = variable._binning_table_optimal.WoE.values[:-1]
            elif self._target_dtype == "numerical":
                if variable.metric == "mean":
                    woe = variable._binning_table_optimal.Mean.values[:-1]
                elif variable.metric == "median":
                    woe = variable._binning_table_optimal.Median.values[:-1]
                else:
                    woe = variable._binning_table_optimal.Mean.values[:-1]

            # initialize woe and groups arrays
            t_woe = np.zeros(n)
            if not only_metric:
                t_grp = np.zeros(n)

            if not n_splits:
                t_woe = np.ones(n) * woe[0]
                if not only_metric:
                    t_grp = np.zeros(n)
                # indexes with NaN in x
                idx_nan = pd.isnull(x)
                # a single category increase n_splits for special and missing
                n_splits = 1

            elif dtype in ("categorical", "nominal"):
                # categorical and nominal variables return groups as 
                # numpy.ndarray objects.
                for _idx, split in enumerate(variable.splits_optimal):
                    mask = (x.isin(split))
                    t_woe[mask] = woe[_idx]
                    if not only_metric:
                        t_grp[mask] = _idx
                # indexes with NaN in x
                idx_nan = pd.isnull(x)

            else:
                # numerical and ordinal variables return extra group
                # (> last split)
                splits = variable.splits_optimal[::-1]
                mask = (x > splits[-1])
                t_woe[mask] = woe[n_splits]
                if not only_metric:
                    t_grp[mask] = n_splits
                for _idx, split in enumerate(splits):
                    mask = (x <= split)
                    t_woe[mask] = woe[n_splits - (_idx + 1)]
                    if not only_metric:
                        t_grp[mask] = n_splits - (_idx + 1)
                # indexes with NaN n x
                idx_nan = pd.isnull(x)
                # account for > group
                n_splits += 1

            # special values
            if variable._splits_specials:
                for _idx, split in enumerate(variable._splits_specials):
                    if isinstance(split, np.ndarray):
                        mask = (x.isin(split))
                    else:
                        mask = (x == split)
                    t_woe[mask] = woe[n_splits]
                    if not only_metric:
                        t_grp[mask] = n_splits
                    n_splits += 1
            else:
                t_woe[idx_special] = woe[n_splits]
                n_splits += 1

            # missing values
            t_woe[idx_nan] = woe[n_splits]
            # check why t_grp[idx_nan] = n_splits is missing

            data[varname] = t_woe
            if not only_metric:
                data[grp_name] = t_grp

        # add target
        if add_target:
            data["target"] = self._target

        # transform to numeric type
        data.apply(pd.to_numeric, errors="ignore")

    def variable_binning_table(self, name):
        """
        Binning table for a given variable.

        Parameters
        ----------
        name : str
            The variable name.

        Returns
        -------
        binning_table : pandas.DataFrame            
        """
        id = self._detect_name_id(name)
        optbin = self._variables_information[id]
        return optbin.binning_table()

    def variable_plot_binning_result(self, name, plot_type="pd",
        plot_bar_type="event"):
        """
        Binning table plot for a given variable.

        Parameters
        ----------
        name : str
            The variable name.

        plot_type : str (default="pd")
            The measure to show in y-axis. Options are: "pd" and "woe" for
            binary target.

        plot_bar_type : str (default="event")
            The count value to show in barplot. Options are: "all", "event"
            and "nonevent" for binary target.
        """
        id = self._detect_name_id(name)
        optbin = self._variables_information[id]
        if self._target_dtype == "binary":
            return optbin.plot_binning_table(plot_type, plot_bar_type)
        else:
            return optbin.plot_binning_table()

    def variable_plot_optimizer_progress(self, name):
        """
        Optimization solver progress for a given variable.

        Parameters
        ----------
        name : str
            The variable name.
        """
        id = self._detect_name_id(name)
        optbin = self._variables_information[id]
        return optbin.plot_optimizer_progress()

    def variable_splits(self, name):
        """
        Optimal splits returned by OptBin for a given variable.

        Parameters
        ----------
        name : str
            The variable name.

        Returns
        -------
        splits : numpy.ndarray          
        """
        id = self._detect_name_id(name)
        optbin = self._variables_information[id]
        return optbin.splits_optimal

    def variable_stats(self, name):
        """
        OptBin statistics for a given variable.

        Parameters
        ----------
        name : str
            The variable name.
        """
        id = self._detect_name_id(name)
        optbin = self._variables_information[id]
        return optbin.stats()

    def _detect_name_id(self, name):
        """
        Check if variable name in dataframe. Optimize search to stop when 
        occurrence, one should not expect repeated names.
        """
        if not self._is_run:
            raise NotRunException(self)

        if not name in self._column_names:
            raise ValueError("variable {} not in dataframe.".format(name))

        return next(i for (i, v) in enumerate(self._variables_information
            ) if v.name == name)

    def _run(self, data, sample_weight=None):
        """Run optimal grouping + cheks."""
        time_run = time.perf_counter()      

        # perform preliminary operations to extract target
        self._setup(data)

        # run OptBin algorithm for each variable in dataset
        if self.verbose:
            print("running OptimalGrouping...")

        for id, name in enumerate(self._column_names):
            if self.verbose:
                print("\nvariable {}: {}".format(id, name))
            self._variables_information.append(
                self._run_variable(data, name, sample_weight))

        # statistics
        self._statistics()

        self._time_run = time.perf_counter() - time_run

        # update flag
        self._is_run = True

        return self

    def _run_variable(self, data, name, sample_weight=None):
        """
        Run Optimal binning optimizer (OptBin) for a given variable using 
        default settings. 

        Note: for numerical and ordinal variable types OptBin uses RTree as 
        a prebinning algorithm and CTree otherwise.
        """

        # variable type
        x = data[name].values
        dtype = check_dtype(name, x.dtype, self.variables_nominal, 
            self.verbose)

        if self.verbose:
            print("variable dtype:", dtype)

        # instantiate and configure OptBin
        if self._target_dtype == "binary":
            if dtype in ("categorical", "nominal"):
                optbin_solver = OptBin(name=name,
                    dtype=dtype,
                    prebinning_algorithm="rtree",
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
                    metric=self.continuous_metric,
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
                    metric=self.continuous_metric,
                    monotonicity_force=self.monotonicity_force,
                    special_values=self.special_values,
                    special_handler_policy=self.special_handler_policy,
                    special_woe_policy=self.special_woe_policy,
                    missing_woe_policy=self.missing_woe_policy,
                    verbose=self.verbose)

        if self.optbin_options is not None:
            if name in self.optbin_options.keys():
                if self._target_dtype == "binary":
                    # run with default parameters and save results
                    optbin_solver_default = deepcopy(optbin_solver)

                    optbin_solver_default.fit(x, self._target, sample_weight)

                    d_iv = optbin_solver_default._iv_optimal
                    d_n_buckets = optbin_solver_default._n_optimal_buckets
                    d_monotonicity_sense = optbin_solver_default.monotonicity_sense
                    d_largest_bucket = optbin_solver_default._largest_bucket_perc
                    d_smallest_bucket = optbin_solver_default._smallest_bucket_perc

                    self._dict_optbin_options[name] = {
                        "iv": d_iv,
                        "n_buckets": d_n_buckets,
                        "monotonicity_sense": d_monotonicity_sense,
                        "largest_bucket": d_largest_bucket,
                        "smallest_bucket": d_smallest_bucket
                    }
                elif self._target_dtype == "numerical":
                    self._dict_optbin_options[name] = {}

                # pass user options and comments
                user_params = self.optbin_options[name]["params"]
                user_comment = self.optbin_options[name]["comment"]
                optbin_solver.set_params(**user_params)

                if self.verbose:
                    print("- parameters were overwritten by user.\n"
                        "- user comment: {}.".format(user_comment))
                    print("- new parameters:")
                    print(user_params)

        optbin_solver.fit(x, self._target, sample_weight)

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

    def _setup(self, data):
        """
        Perform preliminary analysis of the input database and check target 
        column.
        """
        self._n_samples= len(data)
        self._n_vars = len(data.columns)
        self._column_names = list(data.columns)

        # check whether given target column is in dataframe
        if not self.target in self._column_names:
            raise ValueError("target column {} not available in dataframe."
                .format(self.target))

        # select target data. save target to avoid querying several times.
        self._target = data[self.target].values

        # check target type
        self._target_dtype = check_target_dtype(self._target)

        if self.verbose:
            print("target type is {}.".format(self._target_dtype))

        # exclude target from the list of variables
        self._column_names.remove(self.target)

    def _statistics(self):
        """Compute analysis statistics."""
        if self.verbose:
            print("\nOptimalGrouping completed.")
            print("calculating OptimalGrouping statistics ...")
        
        if self._target_dtype == "binary":
            # summary statistics - generate dataframe for posterior analysis
            header = ["variable", "dtype", "IV", "PD_monotonicity", "groups", 
                "group_special", "group_missing", "group_others",
                "smallest_bucket", "smallest_bucket_(%)","largest_bucket", 
                "largest_bucket_(%)", "std_buckets", "max_p_value"]

            fields = [name, dtype, iv, monotonicity, groups, group_special, 
                group_missing, group_others, smallest_bucket_size, 
                smallest_bucket_perc, largest_bucket_size, largest_bucket_perc, 
                std_bucket_size, max_pvalue] = [[] for _ in range(len(header))]

            for variable in self._variables_information:
                name.append(variable.name)
                dtype.append(variable.dtype)
                iv.append(variable._iv_optimal)
                monotonicity.append(variable.monotonicity_sense)
                groups.append(variable._n_optimal_buckets)
                group_special.append(variable._group_special)
                group_missing.append(variable._group_missing)
                group_others.append(int(variable._group_others))
                max_pvalue.append(variable._max_pvalue)
                largest_bucket_perc.append(variable._largest_bucket_perc)
                smallest_bucket_perc.append(variable._smallest_bucket_perc)
                largest_bucket_size.append(variable._largest_bucket_size)
                smallest_bucket_size.append(variable._smallest_bucket_size)
                std_bucket_size.append(variable._std_bucket_size)
        
        elif self._target_dtype == "numerical":
            # summary statistics - generate dataframe for posterior analysis
            header = ["variable", "dtype", "PD_monotonicity", "groups", 
                "group_special", "group_missing", "group_others",
                "smallest_bucket", "smallest_bucket_(%)","largest_bucket", 
                "largest_bucket_(%)", "std_buckets"]

            fields = [name, dtype, monotonicity, groups, group_special, 
                group_missing, group_others, smallest_bucket_size, 
                smallest_bucket_perc, largest_bucket_size, largest_bucket_perc, 
                std_bucket_size] = [[] for _ in range(len(header))]

            for variable in self._variables_information:
                name.append(variable.name)
                dtype.append(variable.dtype)
                monotonicity.append(variable.monotonicity_sense)
                groups.append(variable._n_optimal_buckets)
                group_special.append(variable._group_special)
                group_missing.append(variable._group_missing)
                group_others.append(int(variable._group_others))
                largest_bucket_perc.append(variable._largest_bucket_perc)
                smallest_bucket_perc.append(variable._smallest_bucket_perc)
                largest_bucket_size.append(variable._largest_bucket_size)
                smallest_bucket_size.append(variable._smallest_bucket_size)
                std_bucket_size.append(variable._std_bucket_size)

        self._summary_table = pd.DataFrame(dict(zip(header, fields)), 
            columns=header)

        # # count occurrences
        monotonicity_dtypes = [optbin.monotonicity_sense for optbin in 
            self._variables_information]

        self._n_vars_ascending = monotonicity_dtypes.count("ascending")
        self._n_vars_descending = monotonicity_dtypes.count("descending") 
        self._n_vars_peak = monotonicity_dtypes.count("peak")
        self._n_vars_valley = monotonicity_dtypes.count("valley")
        self._n_vars_undefined = monotonicity_dtypes.count("undefined")

        # IV analysis
        if self._target_dtype == "binary":
            iv = self._summary_table.IV
            self._iv_min = iv.min()
            self._iv_max = iv.max()
            self._iv_mean = iv.mean()
            self._iv_median = iv.median()

            self._iv_small = len(iv[iv < 0.05])
            self._iv_large = len(iv[iv > 0.50])

            self._iv_small_perc = self._iv_small / self._n_vars
            self._iv_large_perc = self._iv_large / self._n_vars

            # p-value
            pvalue = self._summary_table.max_p_value
            self._pvalue_min = pvalue.min()
            self._pvalue_max = pvalue.max()

            self._pvalue_small = len(pvalue[pvalue <= 0.05])
            self._pvalue_large = len(pvalue[pvalue > 0.05])
            self._pvalue_small_perc = self._pvalue_small / self._n_vars
            self._pvalue_large_perc = self._pvalue_large / self._n_vars

        # monotonicity
        self._pd_asc_perc = self._n_vars_ascending / self._n_vars
        self._pd_desc_perc = self._n_vars_descending / self._n_vars
        self._pd_peak_perc = self._n_vars_peak / self._n_vars
        self._pd_valley_perc = self._n_vars_valley / self._n_vars
        self._pd_undefined_perc = self._n_vars_undefined / self._n_vars

        # Groups
        groups = self._summary_table.groups
        self._groups_max = groups.max()
        self._groups_min = groups.min()
        self._groups_0 = len(groups[groups == 0])
        self._groups_1 = len(groups[groups == 1])
        # multiple specials are possible
        self._groups_special = len(self._summary_table[
            self._summary_table.group_special != 0])
        self._groups_missing = len(self._summary_table[
            self._summary_table.group_missing == 1])
        self._groups_others = len(self._summary_table[
            self._summary_table.group_others == 1])

        self._groups_0_perc = self._groups_0 / self._n_vars
        self._groups_1_perc = self._groups_1 / self._n_vars
        self._groups_missing_perc = self._groups_missing / self._n_vars
        self._groups_special_perc = self._groups_special / self._n_vars
        self._groups_others_perc = self._groups_others / self._n_vars

    def _stats_report(self):
        """
        Generate extended report. 

        Note: this should include most of the metrics required by the summary 
        in the GRMlab report section.
        """

        # general information
        spec_values = "yes" if self.special_values else "no"
        nominal_vars = "yes" if self.variables_nominal else "no"

        # timing
        time_numerical_perc = self._time_run_numerical / self._time_run
        time_ordinal_perc = self._time_run_ordinal / self._time_run
        time_categorical_perc = self._time_run_categorical / self._time_run
        time_nominal_perc = self._time_run_nominal / self._time_run

        if self._target_dtype == "binary":
            report = (
            "\033[94m================================================================================\033[0m\n"
            "\033[1m\033[94m             GRMlab OptimalGrouping 0.1: Automatic Optimal Grouping             \033[0m\n"
            "\033[94m================================================================================\033[0m\n"
            "\n"
            " \033[1mGeneral information                      Configuration options\033[0m\n"
            "   number of records   {:>8}             special values                {:>3}\n"
            "   number of variables {:>8}             nominal variables             {:>3}\n"                                                        
            "   target variable     {:>8}             special handler policy {:>10}\n"
            "   target type         {:>8}             special WoE policy     {:>10}\n"
            "   numerical           {:>8}             missing WoE policy     {:>10}\n"
            "   ordinal             {:>8}\n"
            "   categorical         {:>8}\n"
            "   nominal             {:>8}\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            "\n"
            " \033[1mIV analysis                              Monotonicity analysis\033[0m\n"
            "   min IV             {:>5.3f}                 PD ascending         {:>4} ({:>5.1%})\n"
            "   max IV             {:>5.3f}                 PD descending        {:>4} ({:>5.1%})\n"
            "   mean IV            {:>5.3f}                 PD peak              {:>4} ({:>5.1%})\n"
            "   median IV          {:>5.3f}                 PD valley            {:>4} ({:>5.1%})\n"
            "   IV < 0.05           {:>4} ({:>5.1%})         PD undefined         {:>4} ({:>5.1%})\n"
            "   IV > 0.50           {:>4} ({:>5.1%})\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            "\n"
            " \033[1mp-value analysis                         Groups analysis\033[0m\n"
            "   min p-value        {:>4.3f}                 min groups           {:>4}\n"
            "   max p-value        {:>4.3f}                 max groups           {:>4}\n"
            "   p-value (<= 0.05)   {:>4} ({:>5.1%})         groups = 0           {:>4} ({:>5.1%})\n"
            "   p-value ( > 0.05)   {:>4} ({:>5.1%})         groups = 1           {:>4} ({:>5.1%})\n"
            "                                            group missing        {:>4} ({:>5.1%})\n"
            "                                            group special        {:>4} ({:>5.1%})\n"
            "                                            group others         {:>4} ({:>5.1%})\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            "\n"
            " \033[1mTiming statistics\033[0m\n"
            "   total             {:>7.3f}\n"
            "     numerical       {:>7.3f} ({:>5.1%})\n"
            "     ordinal         {:>7.3f} ({:>5.1%})\n"
            "     categorical     {:>7.3f} ({:>5.1%})\n"
            "     nominal         {:>7.3f} ({:>5.1%})\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            ).format(self._n_samples, spec_values, self._n_vars, nominal_vars, 
            self.target, self.special_handler_policy, self._target_dtype,
            self.special_woe_policy, self._n_vars_numerical, 
            self.missing_woe_policy, self._n_vars_ordinal, 
            self._n_vars_categorical, self._n_vars_nominal, self._iv_min, 
            self._n_vars_ascending, self._pd_asc_perc, self._iv_max, 
            self._n_vars_descending, self._pd_desc_perc, self._iv_mean, 
            self._n_vars_peak, self._pd_peak_perc, self._iv_median, 
            self._n_vars_valley, self._pd_valley_perc, self._iv_small, 
            self._iv_small_perc, self._n_vars_undefined, self._pd_undefined_perc, 
            self._iv_large, self._iv_large_perc, self._pvalue_min, self._groups_min, 
            self._pvalue_max, self._groups_max, self._pvalue_small, 
            self._pvalue_small_perc, self._groups_0, self._groups_0_perc, 
            self._pvalue_large, self._pvalue_large_perc, self._groups_1, 
            self._groups_1_perc, self._groups_missing, self._groups_missing_perc, 
            self._groups_special, self._groups_special_perc, self._groups_others, 
            self._groups_others_perc, self._time_run, 
            self._time_run_numerical, time_numerical_perc, 
            self._time_run_ordinal, time_ordinal_perc, 
            self._time_run_categorical, time_categorical_perc,
            self._time_run_nominal, time_nominal_perc)

        else:
            report = (
            "\033[94m================================================================================\033[0m\n"
            "\033[1m\033[94m             GRMlab OptimalGrouping 0.1: Automatic Optimal Grouping             \033[0m\n"
            "\033[94m================================================================================\033[0m\n"
            "\n"
            " \033[1mGeneral information                      Configuration options\033[0m\n"
            "   number of records   {:>8}             special values                {:>3}\n"
            "   number of variables {:>8}             nominal variables             {:>3}\n"                                                        
            "   target variable     {:>8}             special handler policy {:>10}\n"
            "   target type         {:>9}             special WoE policy     {:>10}\n"
            "   numerical           {:>8}             missing WoE policy     {:>10}\n"
            "   ordinal             {:>8}\n"
            "   categorical         {:>8}\n"
            "   nominal             {:>8}\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            "\n"
            " \033[1mMonotonicity analysis\033[0m\n"
            "   PD ascending         {:>4} ({:>5.1%})\n"
            "   PD descending        {:>4} ({:>5.1%})\n"
            "   PD peak              {:>4} ({:>5.1%})\n"
            "   PD valley            {:>4} ({:>5.1%})\n"
            "   PD undefined         {:>4} ({:>5.1%})\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            "\n"
            " \033[1mGroups analysis\033[0m\n"
            "   min groups           {:>4}\n"
            "   max groups           {:>4}\n"
            "   groups = 0           {:>4} ({:>5.1%})\n"
            "   groups = 1           {:>4} ({:>5.1%})\n"
            "   group missing        {:>4} ({:>5.1%})\n"
            "   group special        {:>4} ({:>5.1%})\n"
            "   group others         {:>4} ({:>5.1%})\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            "\n"
            " \033[1mTiming statistics\033[0m\n"
            "   total             {:>7.3f}\n"
            "     numerical       {:>7.3f} ({:>5.1%})\n"
            "     ordinal         {:>7.3f} ({:>5.1%})\n"
            "     categorical     {:>7.3f} ({:>5.1%})\n"
            "     nominal         {:>7.3f} ({:>5.1%})\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            ).format(self._n_samples, spec_values, self._n_vars, nominal_vars, 
            self.target, self.special_handler_policy, self._target_dtype,
            self.special_woe_policy, self._n_vars_numerical, 
            self.missing_woe_policy, self._n_vars_ordinal, 
            self._n_vars_categorical, self._n_vars_nominal,
            self._n_vars_ascending, self._pd_asc_perc,
            self._n_vars_descending, self._pd_desc_perc,
            self._n_vars_peak, self._pd_peak_perc,
            self._n_vars_valley, self._pd_valley_perc,
            self._n_vars_undefined, self._pd_undefined_perc, 
            self._groups_min, 
            self._groups_max,
            self._groups_0, self._groups_0_perc, 
            self._groups_1, 
            self._groups_1_perc, self._groups_missing, self._groups_missing_perc, 
            self._groups_special, self._groups_special_perc, self._groups_others, 
            self._groups_others_perc, self._time_run, 
            self._time_run_numerical, time_numerical_perc, 
            self._time_run_ordinal, time_ordinal_perc, 
            self._time_run_categorical, time_categorical_perc,
            self._time_run_nominal, time_nominal_perc)          

        print(report)

    @property
    def variables_information(self):
        """Return list of OptBin variables class."""
        if not self._is_run:
            raise NotRunException(self)

        return self._variables_information
