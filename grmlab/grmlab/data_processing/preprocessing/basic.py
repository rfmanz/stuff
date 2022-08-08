"""
Data preprocessing
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import gc
import time

import numpy as np
import pandas as pd

from ...core.base import GRMlabProcess
from ...core.dtypes import is_number
from ...core.dtypes import is_numpy_float
from ...core.dtypes import is_numpy_int
from ...core.dtypes import is_numpy_object
from ...core.dtypes import is_string
from ...core.exceptions import NotRunException
from ...reporting.util import reporting_output_format
from .algorithms import connected_pairs
from .algorithms import fast_array_constant
from .algorithms import find_duplicates
from .algorithms import information_blocks
from .util import is_binary_integer
from .util import is_date_integer
from .util import is_date_object
from .util import is_exclude_case
from .util import is_identifier
from .util import is_numeric_conversion


RANDOM_SAMPLES = 25

STATUS_BINARY = "binary"
STATUS_CONSTANT = "constant"
STATUS_DATE = "date"
STATUS_EMPTY = "empty"
STATUS_EXCLUDE = "exclude"
STATUS_ID = "id"
STATUS_NAN_UNIQUE = "nan_unique"
STATUS_NUMERIC_CONVERSION = "numeric_conversion"
STATUS_OK = "ok"
STATUS_SPECIAL = "special"
STATUS_SPECIAL_CONSTANT = "special_constant"
STATUS_SPECIAL_UNIQUE = "special_unique"

STATUS_OPTIONS = [
    STATUS_BINARY, STATUS_CONSTANT, STATUS_DATE, STATUS_EMPTY, STATUS_EXCLUDE,
    STATUS_ID, STATUS_NAN_UNIQUE, STATUS_NUMERIC_CONVERSION, STATUS_OK,
    STATUS_SPECIAL, STATUS_SPECIAL_CONSTANT, STATUS_SPECIAL_UNIQUE]

STATUS_REMOVE = [
    STATUS_CONSTANT, STATUS_EMPTY, STATUS_EXCLUDE, STATUS_SPECIAL,
    STATUS_SPECIAL_CONSTANT, STATUS_SPECIAL_UNIQUE]

STATUS_REVIEW = [
    STATUS_BINARY, STATUS_DATE, STATUS_ID]

STATUS_TRANSFORM = [
    STATUS_NAN_UNIQUE, STATUS_NUMERIC_CONVERSION]

STATUS_VALID = list(set(STATUS_OPTIONS) ^ set(STATUS_REMOVE))


def _preprocessing_results(dict_columns, step, collapsed, format):
    """Return run or transform step results as a dataframe or json."""
    order_cols = [
        "name", "dtype", "block_id", "recommended_action", "constant",
        "empty", "exclude", "special", "special_constant", "special_unique",
        "nan_unique", "numeric_conversion", "binary", "date", "id", "ok",
        "duplicate", "duplicate_of"]

    order_cols_collapsed = [
        "name", "dtype", "block_id", "recommended_action", "status",
        "duplicate", "duplicate_of"]

    order_cols_transform = [
        "name", "dtype", "block_id", "action", "status",
        "comment", "recommended_action", "user_action", "user_comment",
        "auto_comment"]

    if step == "run":
        results = pd.DataFrame.from_dict(dict_columns).T
        results.reset_index(level=0, inplace=True)
        results.rename(columns={"index": "name"}, inplace=True)

        if not collapsed:
            results_expand = pd.get_dummies(results.status)
            add_cols = [
                "name", "recommended_action", "block_id", "dtype",
                "duplicate", "duplicate_of"]
            results_expand[add_cols] = results[add_cols]

            # add empty cols if not dummy was generated
            for col in order_cols:
                if col not in results_expand.columns:
                    results_expand[col] = 0

            results_expand = results_expand[order_cols]
            return reporting_output_format(results_expand, format)
        else:
            results = results[order_cols_collapsed]
            return reporting_output_format(results, format)
    else:
        results = pd.DataFrame.from_dict(dict_columns).T
        results.reset_index(level=0, inplace=True)
        results.rename(columns={"index": "name"}, inplace=True)

        for col in ["comment", "user_action", "user_comment", "auto_comment"]:
            if col not in results.columns:
                results[col] = np.nan

        results = results[order_cols_transform]
        return reporting_output_format(results, format)


def _preprocessing_stats(report_data, step):
    """Return preprocessing statistics report."""
    if step == "run":
        report = (
            "\033[94m================================================================================\033[0m\n"
            "\033[1m\033[94m                         GRMlab Preprocessing 0.1: Run                          \033[0m\n"
            "\033[94m================================================================================\033[0m\n"
            "\n"
            " \033[1mGeneral information                      Configuration options\033[0m\n"
            "   number of samples   {:>8}             special values                {:>3}\n"
            "   number of columns   {:>8}\n"
            "   target column       {:>8}\n"
            "   date column         {:>8}\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            "\n"
            " \033[1mColumn analysis                          Block analysis\033[0m\n"
            "   binary              {:>8}             information blocks           {:>4}\n"
            "   constant            {:>8}             column largest block         {:>4}\n"
            "   date                {:>8}             ungrouped columns            {:>4}\n"
            "   empty               {:>8}\n"
            "   exclude             {:>8}           \033[1mDuplicates\033[0m\n"
            "   id                  {:>8}             duplicates                   {:>4}\n"
            "   nan_unique          {:>8}             duplicates_of                {:>4}\n"
            "   numeric_conversion  {:>8}\n"
            "   ok                  {:>8}\n"
            "   special             {:>8}\n"
            "   special_constant    {:>8}\n"
            "   special_unique      {:>8}\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            "\n"
            " \033[1mTiming statistics\033[0m\n"
            "   total                {:>7.3f}\n"
            "     categorize         {:>7.3f} ({:>5.1%})\n"
            "     duplicates         {:>7.3f} ({:>5.1%})\n"
            "     information blocks {:>7.3f} ({:>5.1%})\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            )
    else:
        report = (
            "\033[94m================================================================================\033[0m\n"
            "\033[1m\033[94m                       GRMlab Preprocessing 0.1: Transform                      \033[0m\n"
            "\033[94m================================================================================\033[0m\n"
            "\n"
            " \033[1mResults                                        Timing statistics\033[0m\n"
            "   original data       {:>8}                   total       {:>7.3f}\n"
            "   after preprocessing {:>8}                     remove    {:>7.3f} ({:>5.1%})\n"
            "   removed             {:>8}                     transform {:>7.3f} ({:>5.1%})\n"
            "   transformed         {:>8}\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            "\n"
            "\033[1mUser actions\033[0m\n"
            "   number of actions   {:>8}\n"
            "   number of comments  {:>8} ({:>5.1%})\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            )

    return report.format(*report_data)


def _run_column(data, n_samples, special_values):
    """Inspect a given column and categorize."""
    idx_special = data.isin(special_values)
    idx_nan = data.isna()
    flag_special = np.any(idx_special)

    clean = data[(~idx_nan) & (~idx_special)].values
    n_clean = clean.shape[0]

    if not n_clean:
        if np.count_nonzero(idx_special) == n_samples:
            idx_rnd = np.random.randint(0, n_samples, RANDOM_SAMPLES)
            if bool(np.asarray(data.values[idx_rnd] == data.values[0]).all()):
                if fast_array_constant(data.values):
                    return STATUS_SPECIAL_CONSTANT
                else:
                    return STATUS_SPECIAL
            else:
                return STATUS_SPECIAL
        else:
            return STATUS_EMPTY
    else:
        idx_rnd = np.random.randint(0, n_clean, RANDOM_SAMPLES)
        if bool(np.asarray(clean[idx_rnd] == clean[0]).all()):
            # detect whether column is constant
            if fast_array_constant(clean):
                if n_clean < n_samples:
                    # data contains a unique value and the rest are
                    # missing or special.
                    if flag_special:
                        return STATUS_SPECIAL_UNIQUE
                    else:
                        return STATUS_NAN_UNIQUE
                else:
                    # constant column
                    return STATUS_CONSTANT

        if is_numpy_float(data.dtype):
            clean_int = clean.astype(int)
            if bool(np.asarray(clean == clean_int).all()):
                if is_binary_integer(clean_int):
                    return STATUS_BINARY
                elif is_date_integer(clean_int):
                    return STATUS_DATE
                elif is_identifier(clean_int, n_samples):
                    return STATUS_ID

        elif is_numpy_int(data.dtype):
            if is_binary_integer(clean):
                return STATUS_BINARY
            elif is_date_integer(clean):
                return STATUS_DATE
            elif is_identifier(clean, n_samples):
                return STATUS_ID

        elif is_numpy_object(data.dtype):
            if is_date_object(clean):
                return STATUS_DATE
            elif is_identifier(clean, n_samples):
                return STATUS_ID
            elif is_numeric_conversion(clean):
                return STATUS_NUMERIC_CONVERSION

        return STATUS_OK


class Preprocessing(GRMlabProcess):
    """
    Data preprocessing.

    Perform data cleaning and transformations to raw dataset. Data cleaning is
    a common requirement that benefits any subsequent modelling step.

    Parameters
    ----------
    target : str or None (default=None)
        The name of the column flagged as target.

    date : str or None (default=None)
        The name of the column flagged as date.

    special_values : list or None (default=None)
        List of special values to be considered.

    verbose : int or boolean (default=False)
        Controls verbosity of output.
    """
    def __init__(self, target=None, date=None, special_values=None,
                 verbose=False):

        self.target = target
        self.date = date
        self.special_values = [] if special_values is None else special_values
        self.verbose = verbose

        self._n_samples = None
        self._n_columns = None
        self._n_columns_remove = None

        self._column_names = []
        self._dict_column = {}
        self._duplicated_columns = []
        self._info_blocks = []
        self._info_blocks_stats = []

        self._time_run = None
        self._time_cases = None
        self._time_duplicates = None
        self._time_info_blocks = None
        self._time_transform = None
        self._time_apply_transform = None
        self._time_remove = None

        # tranform parameters
        self._transform_mode = None

        # flags
        self._is_run = False
        self._is_transformed = False

    def results(self, step="run", collapsed=False, format="dataframe"):
        """
        Return information and flags for each column.

        Parameters
        ----------
        step : str or None (default="run")
            Step name, options are "run" and "transform".

        collapsed : boolean (default=False)
            If collapsed, a compacted view of step "run" is returned. This
            option has no effect if ``step == "transform"``.

        format : str, "dataframe" or "json" (default="dataframe")
            If "dataframe" return pandas.DataFrame. Otherwise, return serialized
            json.
        """
        if step not in ("run", "transform"):
            raise ValueError("step not found.")

        if step is "run" and not self._is_run:
            raise NotRunException(self, "run")
        elif step is "transform" and not self._is_transformed:
            raise NotRunException(self, "transform")

        return _preprocessing_results(self._dict_column, step, collapsed,
                                      format)

    def run(self, data):
        """
        Run preprocessing.

        Preprocessing performs three steps: categorizes each column data, and
        finds duplicates and information blocks.

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

        self._n_samples = len(data)
        self._n_columns = len(data.columns)

        # column names to lowercase
        data.columns = map(str.lower, data.columns)
        self._column_names = list(data.columns)
        if self.target is not None:
            self.target = self.target.lower()
        if self.date is not None:
            self.date = self.date.lower()

        # run preprocessing
        self._run(data)

        self._is_run = True

        return self

    def stats(self, step="run"):
        """
        Preprocessing statistics.

        Parameters
        ----------
        step : str or None (default="run")
            Step name, options are "run" and "transform".
        """
        if step not in ("run", "transform"):
            raise ValueError("step not found.")

        if not self._is_run and step is "run":
            raise NotRunException(self, "run")
        elif not self._is_transformed and step is "transform":
            raise NotRunException(self, "transform")

        dict_values = self._dict_column.values()

        if step == "run":
            special_flag = "yes" if self.special_values else "no"
            target_flag = self.target if self.target is not None else "not set"
            date_flag = self.date if self.date is not None else "not set"

            # categories
            [binary, constant, date, empty, exclude, id, nan_unique,
                numeric_conversion, ok, special, special_constant,
                special_unique] = [sum(
                    [d["status"] == c for d in dict_values])
                    for c in STATUS_OPTIONS]

            # duplicates
            n_duplicates = sum([d["duplicate"] for d in dict_values])
            n_duplicates_of = sum([1 if d["duplicate_of"] is not None else 0
                                   for d in dict_values])

            # information blocks
            n_blocks, largest_block, n_ungrouped = self._info_blocks_stats

            # timing
            perc_time_cases = self._time_cases / self._time_run
            perc_time_duplicates = self._time_duplicates / self._time_run
            perc_time_info_blocks = self._time_info_blocks / self._time_run

            # prepare data
            report_data = [
                self._n_samples, special_flag, self._n_columns,
                target_flag, date_flag, binary, n_blocks, constant,
                largest_block, date, n_ungrouped, empty, exclude, id,
                n_duplicates, nan_unique, n_duplicates_of, numeric_conversion,
                ok, special, special_constant, special_unique, self._time_run,
                self._time_cases, perc_time_cases, self._time_duplicates,
                perc_time_duplicates, self._time_info_blocks,
                perc_time_info_blocks]
        else:
            perc_time_remove = self._time_remove / self._time_transform
            perc_time_transf = self._time_apply_transform / self._time_transform
            n_columns_after = self._n_columns - self._n_columns_remove

            n_columns_transform = sum([d["status"] in STATUS_TRANSFORM
                                       for d in dict_values])

            n_user_actions = sum(1 for d in dict_values if "user_action" in d)
            n_user_comment = sum(d["user_comment"] != "" for d in dict_values
                                 if "user_action" in d)

            if n_user_actions:
                perc_user_comment = n_user_comment / n_user_actions
            else:
                perc_user_comment = 0

            report_data = [
                self._n_columns, self._time_transform,
                n_columns_after, self._time_remove, perc_time_remove,
                self._n_columns_remove, self._time_apply_transform,
                perc_time_transf, n_columns_transform, n_user_actions,
                n_user_comment, perc_user_comment]

        print(_preprocessing_stats(report_data, step))

    def transform(self, data, mode="aggressive"):
        """
        Transform input raw dataset in-place.

        Reduce the raw dataset by removing columns with action flag equal to
        remove.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw dataset.

        mode : str
            Transformation mode, options are "agggresive" and "basic". If
            ``mode=="aggressive"`` columns tagged with action "remove" and
            with status "id" and "date" (except the date column supplied) are
            dropped. If ``mode=="basic"`` only columns tagged as "remove" are
            dropped.

        Returns
        -------
        self : object
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas dataframe.")

        if mode not in ("aggressive", "basic"):
            raise ValueError("mode {} is not supported.".format(mode))

        self._transform_mode = mode

        # apply transformations
        time_init = time.perf_counter()
        self._transform_nan_unique(data)
        self._transform_numeric_conversion(data)
        self._time_apply_transform = time.perf_counter() - time_init

        # remove columns
        self._transform_remove(data, mode=mode)

        self._is_transformed = True
        self._time_transform = self._time_apply_transform + self._time_remove

        return self

    def _run(self, data):
        """Run preprocessing three steps and decision engine."""
        time_init = time.perf_counter()

        # find and tag occurrences
        self._find_cases(data)

        # find duplicates
        self._find_duplicates(data)

        # find information blocks
        self._find_information_blocks(data)

        # set actions
        self._decision_engine()

        self._time_run = time.perf_counter() - time_init

    def _decision_engine(self):
        """Set action to be taken for each column."""
        duplicates_sorted = [
            sorted(list(ls), key=len) for ls in self._duplicated_columns]
        duplicates_all = sum([l for l in duplicates_sorted], [])

        columns = self._column_names
        for column in columns:
            # check whether is duplicate parent or child
            duplicate_remove = 0
            if column in duplicates_all:
                column_block = next(block for block in duplicates_sorted if
                                    column in block)
                parent = column_block[0]
                child = column_block[1:]
                if column in child:
                    duplicate_remove = 1
                    self._dict_column[column]["duplicate_of"] = parent
                self._dict_column[column]["duplicate"] = True

            # set action for each column according to its status
            status = self._dict_column[column]["status"]
            if status in STATUS_REMOVE or duplicate_remove:
                self._dict_column[column]["action"] = "remove"
            elif status in STATUS_REVIEW:
                self._dict_column[column]["action"] = "review"
            elif status in STATUS_TRANSFORM:
                self._dict_column[column]["action"] = "transform"
            else:
                self._dict_column[column]["action"] = "keep"

            action = self._dict_column[column]["action"]
            self._dict_column[column]["recommended_action"] = action

            # check if date and/or target provided by user will be removed
            if self.verbose:
                status_remove = (
                    self._dict_column[self.date]["status"] == "remove")

                if self.date is not None and status_remove:
                    print("date column {} with status {} is flagged with"
                          " action = remove. Check it or change action.")

                if self.target is not None and status_remove:
                    print("target column {} with status {} is flagged with"
                          " action = remove. Check it or change action.")

            # information block id for each column
            id, block = next((id, block) for (id, block) in
                             enumerate(self._info_blocks) if column in block)
            if len(block) > 1:
                self._dict_column[column]["block_id"] = id

    def _find_cases(self, data):
        """Find and tag column occurrences."""
        time_init = time.perf_counter()

        dict_column_info = {
            "action": None,
            "block_id": -1,
            "duplicate": False,
            "duplicate_of": None,
            "status": None,
            "dtype": None}

        # initialize dict of columns
        columns = self._column_names
        grp_dtypes = pd.Index(columns).to_series().groupby(data.dtypes).groups
        for column in columns:
            self._dict_column[column] = dict_column_info.copy()

        # datatypes
        for block_dtype, block_columns in grp_dtypes.items():
            if is_numpy_int(block_dtype):
                dtype = "integer"
            elif is_numpy_float(block_dtype):
                dtype = "float"
            else:
                dtype = "object"
            for column in block_columns:
                self._dict_column[column]["dtype"] = dtype

        # remove columns matching excluded from list of columns to inspect
        exclude_columns = is_exclude_case(columns)
        for column in exclude_columns:
            self._dict_column[column]["status"] = STATUS_EXCLUDE

        for column in list(set(columns) ^ set(exclude_columns)):
            self._dict_column[column]["status"] = _run_column(
                data[column], self._n_samples, self.special_values)

        self._time_cases = time.perf_counter() - time_init

    def _find_duplicates(self, data):
        """Find duplicates."""
        time_init = time.perf_counter()

        columns = [key for key, info in self._dict_column.items()
                   if info["status"] in STATUS_VALID]

        # group columns by datatype
        grp_dtypes = pd.Index(columns).to_series().groupby(data.dtypes).groups

        # search duplicates for each datatype block
        duplicates = []
        for dtype, columns in grp_dtypes.items():
            dtype_data = data[columns].values
            dtype_names = columns.values
            duplicates.append(find_duplicates(dtype, dtype_names, dtype_data))

        # solve disjoint set problem to join
        self._duplicated_columns = connected_pairs(sum(duplicates, []))

        self._time_duplicates = time.perf_counter() - time_init

    def _find_information_blocks(self, data):
        """Find information blocks"""
        time_init = time.perf_counter()
        self._info_blocks, self._info_blocks_stats = information_blocks(data)
        self._time_info_blocks = time.perf_counter() - time_init

    def _transform_nan_unique(self, data):
        """
        Return value to use as a replacement for columns tagged as nan_unique.
        """
        maxlength = 8
        msg = "NaN values replaced by {}"

        for column in data.columns:
            if self._dict_column[column]["status"] == "nan_unique":
                idx_nan = data[column].isna()
                value = data[column][~idx_nan].values[0]

                if value in [0, 1]:
                    replace = int(not value)
                elif is_number(value):
                    replace = 0
                elif is_string(value):
                    if value in ["Y", "S"]:
                        replace = "N"
                    elif value == "N":
                        replace = "Y"
                    elif len(value) > maxlength:
                        replace = "NO_INFO"
                    else:
                        replace = "NO_" + value

                # replace NaN values, in-place operation
                data[column].fillna(value=replace, inplace=True)

                # add comment
                self._dict_column[column]["comment"] = msg.format(replace)

        # garbage collector
        gc.collect()

    def _transform_numeric_conversion(self, data):
        """Transform string number to numeric."""
        for column in data.columns:
            if self._dict_column[column]["status"] == "numeric_conversion":
                data[column] = pd.to_numeric(
                    data[column].str.replace(',', '.'))

        # garbage collector
        gc.collect()

    def _transform_remove(self, data, mode):
        """Remove unnecessary column in-place."""
        AGGR_COMMENT = "auto: remove due to aggressive mode"
        BASIC_COMMENT = "auto: keep due to basic mode"
        DATE_COMMENT = "auto: keep date column"
        TARGET_COMMENT = "auto: keep target column"

        time_init = time.perf_counter()
        remove = [column for column in self._column_names if
                  self._dict_column[column]["action"] == "remove"]

        review = [column for column in self._column_names if
                  self._dict_column[column]["action"] == "review"]

        if mode == "aggressive":
            if self.verbose:
                print("columns with action='remove' or action='review' "
                      "except the date and target column if supplied are "
                      "dropped.")

            for column in review:
                if self.date == column:
                    self._dict_column[column]["action"] = "keep"
                    self._dict_column[column]["auto_comment"] = DATE_COMMENT
                elif self.target == column:
                    self._dict_column[column]["action"] = "keep"
                    self._dict_column[column]["auto_comment"] = TARGET_COMMENT
                else:
                    remove.append(column)
                    self._dict_column[column]["action"] = "remove"
                    self._dict_column[column]["auto_comment"] = AGGR_COMMENT

        elif mode == "basic":
            if self.verbose:
                print("only columns with action='remove' are dropped.")

            # if recommended action is not removed then keep
            for column in review:
                if self.date == column:
                    self._dict_column[column]["auto_comment"] = DATE_COMMENT
                elif self.target == column:
                    self._dict_column[column]["auto_comment"] = TARGET_COMMENT
                else:
                    self._dict_column[column]["auto_comment"] = BASIC_COMMENT

                self._dict_column[column]["action"] = "keep"

        self._n_columns_remove = len(remove)

        # drop column in-place
        data.drop(remove, axis=1, inplace=True)

        # garbage collector
        gc.collect()

        self._time_remove = time.perf_counter() - time_init

    def get_column_status(self, column):
        """
        Get status of the column.

        Parameters
        ----------
        column: str
            The column name.

        Returns
        -------
        status : str
            The status assigned to the column.
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        if column not in self._dict_column.keys():
            raise ValueError("column {} not in data.".format(column))
        else:
            return self._dict_column[column]["status"]

    def set_column_status(self, column, status):
        """
        Set status of the column.

        Parameters
        ----------
        column: str
            The column name.

        status: str
            The status to be set.
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        if column not in self._dict_column.keys():
            raise ValueError("column {} not in data.".format(column))

        if status not in STATUS_OPTIONS:
            raise ValueError("status {} not supported.".format(status))

        self._dict_column[column]["status"] = status

    def get_column_action(self, column):
        """
        Get action applied to the column.

        Parameters
        ----------
        column: str
            The column name.

        Returns
        -------
        action : str
            The action assigned to the column.
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        if column not in self._dict_column.keys():
            raise ValueError("column {} not in data.".format(column))
        else:
            return self._dict_column[column]["action"]

    def set_column_action(self, column, action, comment=""):
        """
        Set action applied to the column.

        Parameters
        ----------
        column: str
            The column name.

        action: str
            The action to be set.
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        if column not in self._dict_column.keys():
            raise ValueError("column {} not in data.".format(column))

        if action not in ("keep", "remove"):
            raise ValueError("action {} not supported.".format(action))

        self._dict_column[column]["user_action"] = action
        self._dict_column[column]["user_comment"] = comment

        self._dict_column[column]["action"] = action
