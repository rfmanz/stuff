"""
Univariate analysis by columns.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import numbers
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..core.dtypes import check_date_format
from ..core.dtypes import check_target_dtype
from .univariate import Univariate


class UnivariateByColumns(Univariate):
    """
    Univariate data analysis.

    Perform basic exploratory data analysis on a dataset one column at a time.
    This allows the analysis of larger datasets.

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

    max_p_missing : float (default=0.99)
        Threshold maximum percentage of missing values per variable.

    max_p_special : float (default=0.99)
        Threshold maximum percentage of special values per variable.

    max_divergence : float (default=0.2)
        Threshold maximum divergence measure value.

    verbose : int or boolean (default=False)
        Controls verbosity of output.

    See also
    --------
    Univariate
    """
    def __init__(self, target=None, date=None, variables_nominal=None,
                 special_values=None, max_p_missing=0.99, max_p_special=0.99,
                 max_divergence=0.2, verbose=False):

        super().__init__(target, date, variables_nominal, special_values,
                         max_p_missing, max_p_special, max_divergence, verbose)

        self.path_csv = None
        self.sep_csv = None
        self.path_parquet = None

    def run(self, path, format, sep_csv=","):
        """
        Run univariate data analysis reading one column at a time.

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
        if not isinstance(self.max_p_missing, numbers.Number) or (
                self.max_p_missing < 0.0 or self.max_p_missing > 1.0):
            raise ValueError("max_p_missing must be a positive number "
                             "in [0, 1].".format(self.max_p_missing))

        if not isinstance(self.max_p_special, numbers.Number) or (
                self.max_p_special < 0.0 or self.max_p_special > 1.0):
            raise ValueError("max_p_special must be a positive number "
                             "in [0, 1].".format(self.max_p_special))

        if not isinstance(self.max_divergence, numbers.Number) or (
                self.max_divergence < 0.0 or self.max_divergence > 1.0):
            raise ValueError("max_divergence must be a positive number "
                             "in [0, 1].".format(self.max_divergence))

        if format not in ("csv", "parquet"):
            raise ValueError("file format {} is not supported.".format(format))

        if format == "csv":
            self.path_csv = path
            self.sep_csv = sep_csv
            self._setup_csv()
        elif format == "parquet":
            self.path_parquet = path
            self._setup_parquet()

        # run univariate
        self._run(file_format=format)

    def transform(self, path, mode="aggressive"):
        """
        Transform input dataset and create transformed file.

        Reduce the raw dataset by removing variables with action flag equal to
        remove.

        Parameters
        ----------
        path : str
            The string path to the transformed CSV file.

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
                           engine='c', encoding='latin-1', low_memory=False,
                           memory_map=True, usecols=[column]).iloc[:, 0]

    def _read_column_parquet(self, column):
        """Read an specific column from Parquet file."""
        return pq.read_table(source=self.path_parquet,
                             columns=[column]).to_pandas().iloc[:, 0]

    def _read_header_csv(self):
        """Read CSV header."""
        return list(pd.read_csv(filepath_or_buffer=self.path_csv,
                    sep=self.sep_csv, engine='c', encoding='latin-1',
                    low_memory=False, memory_map=True, header=None,
                    nrows=1).values[0])

    def _run(self, file_format):
        """Run univariate and decision engine."""
        time_init = time.perf_counter()

        if self.verbose:
            print("running univariate analysis...")

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

    def _setup_csv(self):
        """
        Perform preliminary analysis of the input dataset and check target
        and date column.
        """
        self._column_names = self._read_header_csv()
        self._n_vars = len(self._column_names)

        # check whether given target and date are in dataframe
        if self.target is not None and self.target not in self._column_names:
            raise ValueError("target variable {} not available in dataframe."
                             .format(self.target))

        if self.date is not None and self.date not in self._column_names:
            raise ValueError("date variable {} not available in dataframe."
                             .format(self.date))

        # select target data, save target to avoid querying several times
        if self.target is not None:
            self._target = self._read_column_csv(self.target).values
            self._target_dtype = check_target_dtype(self._target)
            self._column_names.remove(self.target)
            self._n_samples = len(self._target)

        if self.date is not None:
            self._dates = self._read_column_csv(self.date).values
            check_date_format(self._dates)

            self._unique_dates = np.unique(self._dates)
            self._column_names.remove(self.date)
            self._n_samples = len(self._dates)

    def _setup_parquet(self):
        """
        Perform preliminary analysis of the input dataset and check target and
        date column.
        """
        file = pq.ParquetFile(source=self.path_parquet)
        self._column_names = file.schema.names
        self._n_samples = file.metadata.num_rows
        self._n_vars = len(self._column_names)

        # check whether given target and date are in dataframe
        if self.target is not None and self.target not in self._column_names:
            raise ValueError("target column {} not available in dataframe."
                             .format(self.target))

        if self.date is not None and self.date not in self._column_names:
            raise ValueError("date column {} not available in dataframe."
                             .format(self.date))

        # select target data, save target to avoid querying several times
        if self.target is not None:
            self._target = self._read_column_parquet(self.target).values
            self._target_dtype = check_target_dtype(self._target)
            self._column_names.remove(self.target)

        if self.date is not None:
            self._dates = self._read_column_parquet(self.date).values
            check_date_format(self._dates)

            self._unique_dates = np.unique(self._dates)
            self._column_names.remove(self.date)
