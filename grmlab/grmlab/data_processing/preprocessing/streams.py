"""
Data preprocessing with data streams.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import gc
import os
import time

import numpy as np
import pandas as pd

from ...core.base import GRMlabVariable
from ...core.dtypes import is_numpy_float
from ...core.dtypes import is_numpy_int
from ...core.dtypes import is_numpy_object
from ...core.exceptions import NotRunException
from .algorithms import connected_pairs
from .algorithms import fast_array_constant
from .algorithms import find_duplicates
from .algorithms import HyperLogLog
from .basic import Preprocessing
from .basic import STATUS_BINARY
from .basic import STATUS_CONSTANT
from .basic import STATUS_DATE
from .basic import STATUS_EMPTY
from .basic import STATUS_EXCLUDE
from .basic import STATUS_ID
from .basic import STATUS_NAN_UNIQUE
from .basic import STATUS_NUMERIC_CONVERSION
from .basic import STATUS_OK
from .basic import STATUS_SPECIAL
from .basic import STATUS_SPECIAL_CONSTANT
from .basic import STATUS_SPECIAL_UNIQUE
from .basic import STATUS_OPTIONS
from .basic import STATUS_REMOVE
from .basic import STATUS_REVIEW
from .basic import STATUS_TRANSFORM
from .basic import STATUS_VALID
from .util import is_binary_integer
from .util import is_date_integer
from .util import is_date_object
from .util import is_exclude_case
from .util import is_numeric_conversion


def _preprocessing_stats(report_data, step):
    """Return preprocessing statistics report."""
    if step is "run":
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
            "   file size           {:>8.2f} GB\n"
            "   number of chunks    {:>8}\n"
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
            "   total               {:>8.3f}\n"
            "     map operation     {:>8.3f} ({:>5.1%})\n"
            "     reduce operation  {:>8.3f} ({:>5.1%})\n"
            "     disk I/O          {:>8.3f} ({:>5.1%})\n"
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
            )

    return report.format(*report_data)


def run_column(column, chunk, special_values, hll_p=14):
    """Inspect a given column and categorize."""
    if column.exclude:
        idx_missing = chunk.isna()
    else:
        # data type
        if is_numpy_int(chunk.dtype):
            dtype = "integer"
        elif is_numpy_float(chunk.dtype):
            dtype = "float"
        else:
            dtype = "object"

        column.update_dtype(dtype)

        # start
        idx_special = chunk.isin(special_values)
        idx_missing = chunk.isna()
        flag_special = np.any(idx_special)

        # unique special counter
        hll_special = HyperLogLog(p=hll_p)
        hll_special.add(chunk[idx_special].unique())
        column.hll_special.merge(hll_special)

        column.n_missing += np.count_nonzero(idx_missing)
        column.n_special += np.count_nonzero(idx_special)

        # chunk w/o missing and special values
        clean = chunk[(~idx_missing) & (~idx_special)].values
        n_clean = clean.shape[0]

        n_samples = len(chunk)

        if not n_clean:
            if np.count_nonzero(idx_special) == n_samples:
                if fast_array_constant(chunk.values):
                    column.count_special_constant += 1
                else:
                    column.count_special += 1
            else:
                column.count_empty += 1
        else:
            if fast_array_constant(clean):
                if n_clean < n_samples:
                    if flag_special:
                        column.count_special_unique += 1
                    else:
                        column.count_nan_unique += 1
                else:
                    column.count_constant += 1

            elif is_numpy_float(chunk.dtype):
                clean_int = clean.astype(np.int)
                if bool(np.asarray(clean == clean_int).all()):
                    column.min = min(column.min, np.min(clean_int))
                    column.max = max(column.max, np.max(clean_int))
                    column.id_integer = True

                    if is_binary_integer(clean_int):
                        column.count_binary += 1
                    elif is_date_integer(clean_int):
                        column.count_date += 1
                    else:
                        column.count_ok += 1
                else:
                    column.count_ok += 1

            elif is_numpy_int(chunk.dtype):
                column.min = min(column.min, np.min(clean))
                column.max = max(column.max, np.max(clean))
                column.id_integer = True

                if is_binary_integer(clean):
                    column.count_binary += 1
                elif is_date_integer(clean):
                    column.count_date += 1
                else:
                    column.count_ok += 1

            elif is_numpy_object(chunk.dtype):
                if is_date_object(clean):
                    column.count_date += 1
                elif is_numeric_conversion(clean):
                    column.count_numeric_conversion += 1
                else:
                    column.count_ok += 1

            # unique values counter
            hll = HyperLogLog(p=hll_p)
            hll.add(pd.Series(clean).unique())
            column.hll.merge(hll)

    # information blocks
    column.sum_block += np.packbits(idx_missing).sum()


def reduce_status(column, n_chunks, n_samples):
    """Collect results and set column status."""
    other = [
        column.count_nan_unique, column.count_constant, column.count_empty,
        column.count_special, column.count_special_constant,
        column.count_special_unique]

    n_clean = n_samples - (column.n_missing + column.n_special)

    if column.exclude:
        column.status = STATUS_EXCLUDE

    elif column.count_empty == n_chunks:
        column.status = STATUS_EMPTY

    elif column.count_constant == n_chunks and column.unique_count() == 1:
        column.status = STATUS_CONSTANT

    elif sum([column.count_nan_unique, column.count_constant,
             column.count_empty]) == n_chunks and column.unique_count() == 1:
        column.status = STATUS_NAN_UNIQUE

    elif (sum(other) + column.count_numeric_conversion == n_chunks and
          column.count_numeric_conversion >= 1):
        column.status = STATUS_NUMERIC_CONVERSION

    elif (column.count_special_constant == n_chunks and
          column.unique_special_count() == 1):
        column.status = STATUS_SPECIAL_CONSTANT

    elif (sum([column.count_special_unique, column.count_special_constant,
               column.count_constant]) == n_chunks
            and column.unique_special_count() == 1):
        column.status = STATUS_SPECIAL_UNIQUE

    elif sum([column.count_special, column.count_special_constant,
              column.count_constant]) == n_chunks:
        column.status = STATUS_SPECIAL

    elif sum(other) + column.count_date == n_chunks and column.count_date >= 1:
        column.status = STATUS_DATE

    elif (column.unique_count() > 0.9 * n_clean and column.unique_count() > 2
          and n_clean > 0.1 * n_samples and
          (column.dtype is "object" or column.id_integer)):
        column.status = STATUS_ID

    elif (sum(other) + column.count_binary == n_chunks and column.min == 0 and
          column.max == 1 and column.unique_count() == 2):
        column.status = STATUS_BINARY

    else:
        column.status = STATUS_OK

    # duplicates
    if column.count_duplicate == n_chunks:
        column.duplicate = True
    else:
        column.duplicate_of = None


def _print_chunk_info(id, chunk_samples, memory, samples, time_chunk):
    """Print preprocessing chunk information."""
    output = (
        "chunk id {:>5}  -  total processed samples {:>10}\n"
        "-----------------------------------------------------\n"
        "chunk samples            {:>10}\n"
        "chunk (MB)               {:>10.2f}\n"
        "chunk time (s)           {:>10.2f}\n"
        "-----------------------------------------------------"
        ).format(id, samples, chunk_samples, memory, time_chunk)
    print(output)


def _print_file_info(path, file_size):
    """Print preprocessing header."""
    output = (
        "========================================\n"
        "     GRMlab PreprocessingDataStream     \n"
        "========================================\n"
        "file_path       {:>24}\n"
        "file size (GB)  {:>24.3f}\n"
        "========================================\n"
        ).format(path, file_size)

    print(output)


class PreprocessingColumn(GRMlabVariable):
    """
    Data preprocessing column.

    Column information to be updated after a data stream is processed.

    Parameters
    ----------
    name : str
        The name of the column.

    dtype: str
        The data type of the column.

    hll_p: int (default=14)
        It sets the registers size (m) of the HLL data: m=2^p.
    """
    def __init__(self, name, dtype, hll_p=14):
        self.name = name
        self.dtype = dtype
        self.hll_p = hll_p

        # count missing and special values
        self.n_missing = 0
        self.n_special = 0

        # hyperloglog data structures
        self.hll = HyperLogLog(p=self.hll_p)
        self.hll_special = HyperLogLog(p=self.hll_p)

        # dtype flags
        self.mixed_dtypes = False

        # exclude flags
        self.exclude = False

        # duplicate flag
        self.duplicate = False

        # status flag
        self.status = None

        # min / max integer value for binary detection
        self.min = np.inf
        self.max = -np.inf

        # id flag
        self.id_integer = False

        # count flags
        self.count_binary = 0
        self.count_constant = 0
        self.count_date = 0
        self.count_empty = 0
        self.count_ok = 0
        self.count_nan_unique = 0
        self.count_numeric_conversion = 0
        self.count_special = 0
        self.count_special_constant = 0
        self.count_special_unique = 0

        self.count_duplicate = 0
        self.duplicate_of = None

        # info blocks
        self.sum_block = 0

    def unique_count(self):
        """Return number of unique values."""
        return self.hll.count()

    def unique_special_count(self):
        """Return number of unique special values."""
        return self.hll_special.count()

    def update_dtype(self, dtype):
        """Update data type."""
        if dtype != self.dtype:
            self.mixed_dtypes = True

            if dtype is "string":
                self.dtype = "string"
            elif dtype is "float" and self.dtype in ("float", "integer"):
                self.dtype = "float"


class PreprocessingDataStream(Preprocessing):
    """
    Data preprocessing using data streams.

    Perform data cleaning and transformations to raw dataset dividing the
    dataset into several parts (chunks), processing each chunk and merging
    results efficiently.

    Parameters
    ----------
    path : str
        The string path to a CSV file.

    target : str or None (default=None)
        The name of the column flagged as target.

    date : str or None (default=None)
        The name of the column flagged as date.

    special_values : list or None (default=None)
        List of special values to be considered.

    hll_p: int (defaul=14)
        It sets the registers size (m) of the HLL data: m=2^p. 

    verbose : int or boolean (default=False)
        Controls verbosity of output.
    """
    def __init__(self, path, target=None, date=None, special_values=None,
                 hll_p=14, verbose=False):

        self.path = path
        self.target = target
        self.date = date
        self.special_values = [] if special_values is None else special_values
        self.hll_p = hll_p
        self.verbose = verbose

        self._file_size = None

        self._n_samples = None
        self._n_columns = None
        self._n_chunks = None

        self._list_column_objects = []
        self._dict_column = {}

        self._time_run = None
        self._time_map = None
        self._time_reduce = None
        self._time_io = None

        # flags
        self._is_run = False

    def run(self, chunksize, sep=','):
        """
        Run preprocessing.

        Preprocessing performs three steps: categorizes each column data, and
        finds duplicates and information blocks.

        Parameters
        ----------
        chunksize : int
            The number of lines processed for each chunk.

        sep : str
            Delimiter / separator to use when reading a CSV file.

        Returns
        -------
        self : object
        """
        # map operation
        time_chunks = 0
        time_init = time.perf_counter()

        self._file_size = os.path.getsize(self.path) / 1e9
        if self.verbose:
            _print_file_info(self.path, self._file_size)

        for id_chunk, chunk in enumerate(pd.read_csv(
            filepath_or_buffer=self.path, engine='c', sep=sep,
            encoding='latin-1', low_memory=False, chunksize=chunksize,
            memory_map=True)):

            time_init_chunk = time.perf_counter()
            if id_chunk == 0:
                self._setup(chunk)

            self._run_chunk(chunk)
            time_chunk = time.perf_counter() - time_init_chunk
            time_chunks += time_chunk

            if self.verbose:
                memory = chunk.memory_usage(deep=True).sum() / 1e6
                _print_chunk_info(id_chunk, chunk.shape[0], memory,
                                  self._n_samples, time_chunk)

            gc.collect()

        self._time_map = time.perf_counter() - time_init

        # reduce operation
        time_init_reduce = time.perf_counter()
        self._reduce()
        self._time_reduce = time.perf_counter() - time_init_reduce

        # other timings
        self._time_io = self._time_map - time_chunks
        self._time_map = time_chunks
        self._time_run = time.perf_counter() - time_init

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

        dict_values = self._dict_column.values()

        if step is "run":
            special_flag = "yes" if self.special_values else "no"
            target_flag = self.target if self.target is not None else "not set"
            date_flag = self.date if self.date is not None else "not set"

            # categories
            [binary, constant, date, empty, exclude, id, nan_unique,
                numeric_conversion, ok, special, special_constant,
                special_unique] = [sum([d["status"] == c for d in dict_values])
                                   for c in STATUS_OPTIONS]

            # duplicates
            n_duplicates = sum([d["duplicate"] for d in dict_values])
            n_duplicates_of = sum([1 if d["duplicate_of"] is not None else 0
                                   for d in dict_values])

            # information blocks
            n_blocks, largest_block, n_ungrouped = self._info_blocks_stats

            # timing
            perc_time_map = self._time_map / self._time_run
            perc_time_reduce = self._time_reduce / self._time_run
            perc_time_io = self._time_io / self._time_run

            # prepare data
            report_data = [
                self._n_samples, special_flag, self._n_columns,
                target_flag, date_flag, self._file_size, self._n_chunks,
                binary, n_blocks, constant, largest_block, date, n_ungrouped,
                empty, exclude, id, n_duplicates, nan_unique, n_duplicates_of,
                numeric_conversion, ok, special, special_constant,
                special_unique, self._time_run, self._time_map, perc_time_map,
                self._time_reduce, perc_time_reduce, self._time_io,
                perc_time_io]

        print(_preprocessing_stats(report_data, step))

    def _setup(self, chunk):
        """
        Set up preprocessing instantiating each preprocessing column class and
        counters.
        """
        self._n_samples = 0
        self._n_chunks = 0
        self._n_columns = len(chunk.columns)

        chunk.columns = map(str.lower, chunk.columns)
        self.column_names = list(chunk.columns)

        if self.target is not None:
            self.target = self.target.lower()
        if self.date is not None:
            self.date = self.date.lower()

        columns = self.column_names
        grp_dtypes = pd.Index(columns).to_series().groupby(chunk.dtypes).groups

        for block_dtype, block_columns in grp_dtypes.items():
            if is_numpy_int(block_dtype):
                dtype = "integer"
            elif is_numpy_float(block_dtype):
                dtype = "float"
            else:
                dtype = "object"
            for column in block_columns:
                new_column = PreprocessingColumn(column, dtype, self.hll_p)
                self._list_column_objects.append(new_column)

        # exclude columns
        self.exclude_columns = is_exclude_case(self.column_names)
        for column in self._list_column_objects:
            if column.name in self.exclude_columns:
                column.exclude = True

    def _run_chunk(self, chunk):
        """Process a given data chunk."""
        self._n_samples += len(chunk)
        self._n_chunks += 1

        chunk.columns = self.column_names

        # find cases
        for column in self._list_column_objects:
            run_column(column, chunk[column.name], self.special_values,
                       self.hll_p)

        # find duplicates
        valid_columns = list(set(self.column_names) ^ set(self.exclude_columns))
        grp_dtypes = pd.Index(valid_columns).to_series().groupby(
            chunk.dtypes).groups

        duplicates = []
        for dtype, columns in grp_dtypes.items():
            dtype_data = chunk[columns].values
            dtype_names = columns.values
            duplicates.append(find_duplicates(dtype, dtype_names, dtype_data))

        # solve disjoint set problem to join
        duplicate_columns = connected_pairs(sum(duplicates, []))

        duplicates_sorted = [
            sorted(list(ls), key=len) for ls in duplicate_columns]
        duplicates_all = sum([l for l in duplicates_sorted], [])

        for column in self._list_column_objects:
            if column.name in duplicates_all:
                column_block = next(block for block in duplicates_sorted if
                                    column.name in block)
                parent = column_block[0]
                child = column_block[1:]
                if column.name in child:
                    column.duplicate_of = parent
                column.count_duplicate += 1

    def _reduce(self):
        """Collect find cases and find duplicates steps."""
        for column in self._list_column_objects:
            reduce_status(column, self._n_chunks, self._n_samples)

        # find information blocks
        info_column = [(c.name, c.sum_block) for c in self._list_column_objects]

        # generate blocks of candidates to be grouped
        keys = set(list(zip(*info_column))[1])
        groups = [[n for (n, v) in info_column if v == key] for key in keys]

        # sort blocks by size
        info_blocks = sorted(groups, key=len, reverse=True)

        n_blocks = len(info_blocks)
        largest_block = len(info_blocks[0])
        ungrouped = sum(1 for b in info_blocks if len(b) == 1)
        self._info_blocks_stats = [n_blocks-ungrouped, largest_block, ungrouped]

        # generate dict column
        dict_column_info = {
            "action": None,
            "block_id": -1,
            "duplicate": False,
            "duplicate_of": None,
            "status": None,
            "dtype": None}

        for column in self._list_column_objects:
            self._dict_column[column.name] = dict_column_info.copy()
            self._dict_column[column.name]["dtype"] = column.dtype
            self._dict_column[column.name]["status"] = column.status

            # tag as duplicate is valid status
            if column.status in STATUS_VALID:
                self._dict_column[column.name]["duplicate"] = column.duplicate
                if column.duplicate:
                    self._dict_column[
                        column.name]["duplicate_of"] = column.duplicate_of

            # set action
            status = column.status

            if status in STATUS_REMOVE or column.duplicate_of is not None:
                self._dict_column[column.name]["action"] = "remove"
            elif status in STATUS_REVIEW:
                self._dict_column[column.name]["action"] = "review"
            elif status in STATUS_TRANSFORM:
                self._dict_column[column.name]["action"] = "transform"
            else:
                self._dict_column[column.name]["action"] = "keep"

            action = self._dict_column[column.name]["action"]
            self._dict_column[column.name]["recommended_action"] = action

        # check if date and/or target provided by user will be removed
        if self.verbose:
            status_remove = (
                self._dict_column[self.date]["status"] == "remove")

            if self.date is not None and status_remove:
                print("date column {} with status {} is flagged with"
                      " action = remove. Check it or change action.".format(
                        self.date, self._dict_column[self.date]["status"]))

            if self.target is not None and status_remove:
                print("target column {} with status {} is flagged with"
                      " action = remove. Check it or change action.".format(
                        self.target, self._dict_column[self.target]["status"]))

        # information block id for each column
        for id_block, block_columns in enumerate(info_blocks):
            if len(block_columns) > 1:
                for column in block_columns:
                    self._dict_column[column]["block_id"] = id_block

    def transform(self, path, mode="aggressive"):
        """
        Transform input raw dataset a create transformed CSV file.

        Reduce the raw dataset by removing columns with action flag equal to
        remove.

        Parameters
        ----------
        path : str
            The string path to the transformed CSV file.

        mode : str
            Transformation mode, options are "agggresive" and "basic". If
            ``mode=="aggressive"`` columns tagged with action "remove" and
            with status "id" and "date" (except the date column supplied) are
            dropped. If ``mode=="basic"`` only columns tagged as "remove" are
            dropped.

        Returns
        -------
        self : object

        Note
        ----
        Currently not implemented.
        """
        raise NotImplementedError