"""
Data preprocessing with spark.
"""

# Authors: Fernando Gallego Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2019.

import time
                
import numpy as np
import pandas as pd
import pyspark.sql.functions as F

from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType, StringType, DoubleType, DateType

from ....core.dtypes import is_number
from ....core.dtypes import is_string
from ....core.exceptions import NotRunException
from ....data_processing.preprocessing.basic import STATUS_EXCLUDE
from ....data_processing.preprocessing.basic import STATUS_CONSTANT
from ....data_processing.preprocessing.basic import STATUS_BINARY, STATUS_ID
from ....data_processing.preprocessing.basic import STATUS_NUMERIC_CONVERSION
from ....data_processing.preprocessing.basic import STATUS_REMOVE, STATUS_VALID
from ....data_processing.preprocessing.basic import STATUS_EMPTY, STATUS_OK
from ....data_processing.preprocessing.basic import STATUS_SPECIAL, STATUS_DATE
from ....data_processing.preprocessing.basic import STATUS_SPECIAL_UNIQUE
from ....data_processing.preprocessing.basic import STATUS_NAN_UNIQUE
from ....data_processing.preprocessing.basic import STATUS_OPTIONS
from ....data_processing.preprocessing.basic import STATUS_TRANSFORM
from ....data_processing.preprocessing.basic import STATUS_REVIEW
from ....data_processing.preprocessing.basic import Preprocessing
from ....data_processing.preprocessing.util import is_date_integer
from ....data_processing.preprocessing.util import is_date_object
from ....data_processing.preprocessing.util import is_exclude_case
from ....data_processing.preprocessing.util import is_numeric_conversion
from .spark_util import is_sparkSQL_int, is_sparkSQL_float
from .spark_util import decimal_to_float_int
from .spark_util import count_nans_specials_first
from .spark_util import count_nans_specials_first_rdd
from .spark_util import count_distinct, count_constant, integer_float_count
from .spark_util import date_test_object, date_test_int, numeric_conv_test
from .spark_util import find_duplicates_spark_rdd


def _preprocessing_stats(report_data, step):
    """Return preprocessing statistics report."""
    if step is "run":
        report = (
        "\033[94m================================================================================\033[0m\n"
        "\033[1m\033[94m                         GRMlab Preprocessing 0.1: Run                          \033[0m\n"
        "\033[94m================================================================================\033[0m\n"
        "\n"
        " \033[1mGeneral information                      Configuration options\033[0m\n"
        "   number of samples    {:>8}            special values                {:>3}\n"
        "   number of columns    {:>8}\n"
        "   target column        {:>8}\n"
        "   date column          {:>8}\n"
        "   number of partitions {:>8}\n"
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
        "     clean hdfs file   {:>8.3f} ({:>5.1%})\n"
        "     categorize        {:>8.3f} ({:>5.1%})\n"
        "     duplicates        {:>8.3f} ({:>5.1%})\n"
        "   \033[94m--------------------------------------------------------------------------\033[0m\n"
        )
    elif step is "transform":
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


def _update_status(dict_column):
    """Return the columns with any status"""
    return [key for key in dict_column if dict_column[key]["status"] in [
            STATUS_EXCLUDE, STATUS_EMPTY, STATUS_SPECIAL, STATUS_CONSTANT,
            STATUS_SPECIAL_UNIQUE, STATUS_NAN_UNIQUE, STATUS_BINARY,
            STATUS_ID, STATUS_DATE, STATUS_NUMERIC_CONVERSION]]


class PreprocessingSpark(Preprocessing):    
    """
    Data preprocessing using pyspark.

    Perform data cleaning and transformations to raw dataset. Data cleaning is
    a common requirement that benefits any subsequent modelling step.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        HDFS dataframe.

    target : str or None (default=None)
        The name of the column flagged as target.

    date : str or None (default=None)
        The name of the column flagged as date.

    special_values : list or None (default=None)
        List of special values to be considered.

    mapPartitions : boolean (default=False)
        If true it will use mapPartitions transformation, it is faster, but
        more memory consuming. If false, it will use map transformations
        instead, which reads the database record by record; it is slower, but
        it consumes a smaller amount of memory.

    verbose : int or boolean (default=False)
        Controls verbosity of output.
    """

    def __init__(self, df, target=None, date=None, special_values=None,
                 mapPartitions=False, verbose=True):

        # Preprocessing parameters
        self.df = df
        self.special_values = [] if special_values is None else special_values
        self.target = target
        self.date = date
        self.mapPartitions = mapPartitions
        self.verbose = verbose

        # df characteristics
        self._n_partitions = None
        self._n_samples = None
        self._n_columns = None
        self._column_names = []
        self._dict_column = {}
        self._info_blocks_stats = None
        self._n_columns_remove = None

        # time
        self._time_transform = 0
        self._time_apply_transform = 0
        self._time_remove = 0
        self._time_clean_file = 0
        self._time_cases = 0
        self._time_duplicates = 0
        self._time_info_blocks = 0
        self._time_run = 0

        # tranform parameters
        self._transform_mode = None

        # flags
        self._is_run = False
        self._is_transformed = False

    def run(self):
        """
        Run preprocessing.

        Preprocessing performs three steps: categorizes each column data,
        finds duplicates and information blocks
        """
        if not isinstance(self.df, DataFrame):
            raise TypeError("df must be pyspark.sql.DataFrame.")

        time_init_clean = time.perf_counter()
        if self.verbose:
            print("cleaning file....")

        self.df = self._clean_df()
        self._n_partitions = self.df.rdd.getNumPartitions()

        self._time_clean_file = (time.perf_counter() - time_init_clean)
        if self.verbose:
            print(
                ("file loaded - time (s): {:>10.2f}s").format(
                 self._time_clean_file))
            print(
                ("num. partitions: {:>7.0f}\n----").format(
                 self._n_partitions))

        if self.verbose:
            print("evaluation....")

        self._run()

        time_evaluation = (time.perf_counter() - self._time_clean_file -
                           time_init_clean)

        self._time_run = time.perf_counter() - time_init_clean
        if self.verbose:
            print(("evaluation finished - time (s): {:>10.2f}s\n----").format(
                time_evaluation))

        if self.verbose:
            print(
                ("total time run (s): {:>10.2f}\n----").format(self._time_run))

        self._is_run = True

    def stats(self, step="run"):
        """
        Preprocessing statistics.

        Parameters
        ----------
        step : str or None (default="run")
            Step name, options are "run" and "transform".
        """

        dict_values = self._dict_column.values()

        if step == "run":
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
            perc_time_clean = self._time_clean_file / self._time_run
            perc_time_categorize = self._time_cases / self._time_run
            perc_time_duplicates = self._time_duplicates / self._time_run

            # prepare data
            report_data = [
                self._n_samples, special_flag, self._n_columns,
                target_flag, date_flag, self._n_partitions,
                binary, n_blocks, constant, largest_block, date, n_ungrouped,
                empty, exclude, id, n_duplicates, nan_unique, n_duplicates_of,
                numeric_conversion, ok, special, special_constant,
                special_unique, self._time_run,
                self._time_clean_file, perc_time_clean,
                self._time_cases, perc_time_categorize, self._time_duplicates,
                perc_time_duplicates
                ]
            print(_preprocessing_stats(report_data, "run"))

        elif step == "transform":
            # timing
            perc_time_remove = self._time_remove / self._time_transform
            perc_time_transf = self._time_apply_transform/self._time_transform

            # remove and transform
            n_columns_after = self._n_columns - self._n_columns_remove
            n_columns_transform = sum([
                d["status"] in STATUS_TRANSFORM for d in dict_values
                ])

            # prepare data
            report_data = [
                self._n_columns, self._time_transform,
                n_columns_after, self._time_remove, perc_time_remove,
                self._n_columns_remove, self._time_apply_transform,
                perc_time_transf, n_columns_transform
                ]
            print(_preprocessing_stats(report_data, "transform"))

    def transform(self, mode="basic"):
        """
        Transform input raw dataset in-place.

        Reduce the raw dataset by removing columns with action flag equal to
        remove.

        Parameters
        ----------
        mode : str
            Transformation mode, options are "aggressive" and "basic". If
            ``mode=="aggressive"`` columns tagged with action "remove" and
            with status "id" and "date" (except the date column supplied) are
            dropped. If ``mode=="basic"`` only columns tagged as "remove" are
            dropped.
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        if mode not in ("aggressive", "basic"):
            raise ValueError("mode {} is not supported.".format(mode))

        self._transform_mode = mode

        # apply transformations
        time_init = time.perf_counter()
        self._transform_nan_unique()
        self._transform_numeric_conversion()
        self._time_apply_transform = time.perf_counter() - time_init
        if self.verbose:
            print(
                ("transformation finished - time (s): {:>10.2f}s\n----"
                 ).format(self._time_apply_transform)
                )

        # remove columns
        self._transform_remove(mode)
        self._time_transform = (time.perf_counter() - time_init)
        self._time_remove = self._time_transform - self._time_apply_transform
        if self.verbose:
            print(("remove - time (s): {:>10.2f}s\n----").format(
                self._time_remove))

        self._is_transformed = True

    def _clean_df(self):
        """Change to type 'object' the values that are not 'double',
        'int' or 'string' and treat date format."""
        df_clean = self.df
        unique_types = np.unique([elt[1] for elt in df_clean.dtypes])

        # from decimal to float/int
        to_float = []
        to_int = []
        for elt in unique_types:
            type_ = decimal_to_float_int(elt)
            if type_ is None:
                continue
            elif type_ == "float":
                to_float.append(elt)
            elif type_ == "int":
                to_int.append(elt)

        # clean column types
        cols_to_float = []
        cols_to_int = []
        cols_to_str = []
        cols_to_date = []
        cols_rest = []
        for elt in df_clean.dtypes:
            if elt[1] in to_float:
                cols_to_float.append(elt[0])
            elif elt[1] in to_int:
                cols_to_int.append(elt[0])
            elif elt[1] == "date":
                cols_to_str.append(elt[0])
            elif elt[1] == "timestamp":
                cols_to_date.append(elt[0])
            else:
                cols_rest.append(elt[0])

        # apply changes
        df_clean = df_clean.select(
            [F.col(col).cast(DoubleType()) for col in cols_to_float] +
            [F.col(col).cast(IntegerType()) for col in cols_to_int] +
            [F.col(col).cast(StringType()) for col in cols_to_str] +
            [F.col(col).cast(StringType()).cast(DateType()).cast(StringType())
             for col in cols_to_date] +
            cols_rest)

        # Convert to lower case
        df_clean = df_clean.select(
            [F.col(x).alias(x.lower()) for x in df_clean.columns])

        return df_clean

    def _decision_engine(self):
        """Set action to be taken for each column."""

        for column in self._column_names:
            # check whether is duplicate parent or child
            duplicate_remove = False
            if (self._dict_column[column]["duplicate"] and
                    self._dict_column[column]["duplicate_of"] is not None):
                duplicate_remove = True

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
                if (self.date is not None and
                        self._dict_column[self.date]["status"] == "remove"):
                    print(
                        "date column {} with status {} is flagged with"
                        " action = remove. Check it or change action.")

                if (self.target is not None and
                        self._dict_column[self.target]["status"] == "remove"):
                    print(
                        "target column {} with status {} is flagged with"
                        " action = remove. Check it or change action.")

    def _find_cases(self):
        """Find and tag column occurrences."""
        time_init = time.perf_counter()

        dict_column_info = {
            "action": None,
            "recommended_action": None,
            "comment": None,
            "block_id": -1,
            "duplicate": False,
            "duplicate_of": None,
            "status": None,
            "dtype": None,
            "run": None,
        }
        dict_run = {
            "num_specials": 0,
            "num_nans": 0,
            "num_distinct": 0,
            "num_clean_equal": 0,
            "integer_float": False,
            "first_value": None,
            "sum_block": 0
        }

        # initialize dict of columns
        train_types = pd.Series(data=[elt[1] for elt in self.df.dtypes],
                                index=[elt[0] for elt in self.df.dtypes])
        grp_dtypes = pd.Index(self._column_names).to_series(
            ).groupby(train_types).groups
        for col_name in self._column_names:
            self._dict_column[col_name] = dict_column_info.copy()
            self._dict_column[col_name]["run"] = dict_run.copy()

        # datatypes
        for block_dtype, block_columns in grp_dtypes.items():
            if is_sparkSQL_int(block_dtype):
                dtype = "integer"
            elif is_sparkSQL_float(block_dtype):
                dtype = "float"
            else:
                dtype = "object"
            for column in block_columns:
                self._dict_column[column]["dtype"] = dtype

        # excluded cases
        exclude_columns = is_exclude_case(self._column_names)
        for column in exclude_columns:
            self._dict_column[column]["status"] = STATUS_EXCLUDE
        cols_with_status = _update_status(self._dict_column)

        # with mapPartitions. It will need more RAM than without it.
        if self.mapPartitions:
            # count nans, specials and first element.
            time_init = time.perf_counter()
            if self.verbose:
                print("   STATUS_EMPTY/SPECIAL and first detection....")
            non_excluded = [elt for elt in self.df.columns
                            if elt not in cols_with_status]
            [nans_count, special_count,
                first_value, sum_block] = count_nans_specials_first(
                self.df, non_excluded, self.special_values)

            # status empty
            for key in self._dict_column:
                if key not in cols_with_status:
                    self._dict_column[key]["run"]["sum_block"] = sum_block[key]
                    if nans_count[key] == self._n_samples:
                        self._dict_column[key]["status"] = STATUS_EMPTY
                    self._dict_column[key]["run"]["num_nans"] = nans_count[key]

            # status special and empty
            for key in self._dict_column:
                if key not in cols_with_status:
                    self._dict_column[key]["run"][
                        "num_specials"] = special_count[key]
                    if (
                            self._dict_column[key]["run"]["num_nans"] +
                            self._dict_column[key]["run"]["num_specials"]
                            ) == self._n_samples:
                        if self._dict_column[key]["run"]["num_nans"] > 0:
                            # for special values with nans, the empty status
                            # is given
                            self._dict_column[key]["status"] = STATUS_EMPTY
                        else:
                            self._dict_column[key]["status"] = STATUS_SPECIAL
            cols_with_status = _update_status(self._dict_column)

            # first value
            for key in self._dict_column:
                if key not in cols_with_status:
                    # type is needed to detect if the value is a numpy class.
                    if type(first_value[key]).__module__ == "numpy":
                        # .item() is needed to convert from numpy type to
                        # standard Python scalar, otherwise pyspark dataframe
                        # will not recognize the class.
                        self._dict_column[key][
                            "run"]["first_value"] = first_value[key].item()
                    else:
                        self._dict_column[key][
                            "run"]["first_value"] = first_value[key]
            if self.verbose:
                print(("      time (s): {:>10.2f}s").format(
                    time.perf_counter()-time_init))
        # Without mapPartitions. May be slower than with mapPartitions.
        else:
            # count nans, specials and first element.
            time_init = time.perf_counter()
            if self.verbose:
                print("   STATUS_EMPTY/SPECIAL and first detection....")
            non_excluded = [elt for elt in self.df.columns
                            if elt not in cols_with_status]
            [nans_count, special_count,
             first_value, sum_block, _] = count_nans_specials_first_rdd(
                self.df, non_excluded, self.special_values)

            for i, col in enumerate(non_excluded):
                # status empty
                self._dict_column[col]["run"]["sum_block"] = sum_block[i]
                if nans_count[i] == self._n_samples:
                    self._dict_column[col]["status"] = STATUS_EMPTY
                self._dict_column[col]["run"]["num_nans"] = nans_count[i]

                # status special and empty
                self._dict_column[col][
                    "run"]["num_specials"] = special_count[i]
                if (
                        self._dict_column[col]["run"]["num_nans"] +
                        self._dict_column[col]["run"]["num_specials"]
                        ) == self._n_samples:
                    if self._dict_column[col]["run"]["num_nans"] > 0:
                        # for special values with nans, the empty status
                        # is given
                        self._dict_column[col]["status"] = STATUS_EMPTY
                    else:
                        self._dict_column[col]["status"] = STATUS_SPECIAL

                # first value
                self._dict_column[col]["run"]["first_value"] = first_value[i]

            cols_with_status = _update_status(self._dict_column)
            if self.verbose:
                print(("      time (s): {:>10.2f}s").format(
                    time.perf_counter()-time_init))

        # binary-constant candidates
        time_init = time.perf_counter()
        if self.verbose:
            print("   distinct approach....")
        bin_const_candidates = [key for key in self._dict_column
                                if key not in cols_with_status]
        first_approach_distincs = count_distinct(self.df,
                                                 bin_const_candidates, 0.1)
        for elt in bin_const_candidates:
            self._dict_column[elt][
                "run"]["num_distinct"] = first_approach_distincs[0][elt]
        bin_const_candidates = [key for key in bin_const_candidates
                                if first_approach_distincs[0][key] < 3]
        if self.verbose:
            print(("      time (s): {:>10.2f}s").format(
                time.perf_counter()-time_init))

        # status constant, special_unique and nan_unique
        time_init = time.perf_counter()
        if self.verbose:
            print("   STATUS_CONSTANT detection....")
        equal_vals = count_constant(self.df, bin_const_candidates,
                                    self._dict_column)
        for elt in bin_const_candidates:
            self._dict_column[elt][
                "run"]["num_clean_equal"] = equal_vals[0][elt]
            if self._dict_column[elt][
                    "run"]["num_clean_equal"] == self._n_samples:
                self._dict_column[elt]["status"] = STATUS_CONSTANT
            elif (
                    self._dict_column[elt]["run"]["num_clean_equal"] +
                    self._dict_column[elt]["run"]["num_specials"]
                    ) == self._n_samples:
                self._dict_column[elt]["status"] = STATUS_SPECIAL_UNIQUE
            elif (
                    self._dict_column[elt]["run"]["num_clean_equal"] +
                    self._dict_column[elt]["run"]["num_nans"]
                    ) == self._n_samples:
                self._dict_column[elt]["status"] = STATUS_NAN_UNIQUE
            elif (
                    self._dict_column[elt]["run"]["num_clean_equal"] +
                    self._dict_column[elt]["run"]["num_nans"] +
                    self._dict_column[elt]["run"]["num_specials"]
                    ) == self._n_samples:
                self._dict_column[elt]["status"] = STATUS_SPECIAL_UNIQUE
        if self.verbose:
            print(("      time (s): {:>10.2f}s").format(
                time.perf_counter()-time_init))

        # status binary
        time_init = time.perf_counter()
        if self.verbose:
            print("   STATUS_BINARY detection....")
        binary_candidates = [
            elt for elt in bin_const_candidates if (
                self._dict_column[elt]["status"] is None and
                self._dict_column[elt]["run"]["first_value"] in [1, 0])
            ]
        binary_distinc = self.df.select(binary_candidates).distinct().collect()
        for elt in binary_candidates:
            vect = list(
                    filter(
                        lambda x: x is not None, set([
                            binary_distinc[i][elt] for i in range(
                                len(binary_distinc))
                            ])
                        )
                    )
            if len(vect) == 2 and vect[0] in [0, 1] and vect[1] in [0, 1]:
                self._dict_column[elt]["status"] = STATUS_BINARY
        cols_with_status = _update_status(self._dict_column)
        if self.verbose:
            print(("      time (s): {:>10.2f}s").format(
                time.perf_counter()-time_init))

        # integer float check
        time_init = time.perf_counter()
        if self.verbose:
            print("   check if integer float....")
        float_cols = [
            key for key in self._dict_column if (
                self._dict_column[key]["dtype"] == "float" and
                key not in cols_with_status
                )
            ]
        float_check = integer_float_count(self.df, float_cols)
        for elt in float_cols:
            samples = self._n_samples-self._dict_column[elt]["run"]["num_nans"]
            if samples == float_check[0][elt]:
                self._dict_column[elt]["run"]["integer_float"] = True
            else:
                self._dict_column[elt]["status"] = STATUS_OK
        if self.verbose:
            print(("      time (s): {:>10.2f}s").format(
                time.perf_counter()-time_init))

        # status id
        time_init = time.perf_counter()
        if self.verbose:
            print("   STATUS_ID detection....")
        id_candidates = []
        min_samples = 0.1*self._n_samples
        for key in self._dict_column:
            samples = self._n_samples-self._dict_column[key]["run"]["num_nans"]
            # id_candidates are selected from the ones with more than 40% of
            # distinct values. The error for a real 100% disctinc values
            # variable is +/- 20% with 95% confidence interval.
            if (
                    key not in cols_with_status and
                    first_approach_distincs[0][key] > 2 and
                    first_approach_distincs[0][key] > (samples)*0.4):
                if (
                        self._dict_column[key]["dtype"] == "float" and
                        self._dict_column[key]["run"]["integer_float"]):
                    id_candidates.append(key)
                elif self._dict_column[key]["dtype"] != "float":
                    id_candidates.append(key)
        id_check = count_distinct(self.df, id_candidates, 0.02)
        for elt in id_candidates:
            threshold = 0.9
            n_uniques = id_check[0][elt]
            samples = self._n_samples-self._dict_column[elt]["run"]["num_nans"]
            ratio_unique = id_check[0][elt] / (samples)
            min_samples = 0.1*self._n_samples
            if (
                    ratio_unique > threshold and n_uniques > 2 and
                    samples > min_samples):
                self._dict_column[elt]["status"] = STATUS_ID
        cols_with_status = _update_status(self._dict_column)
        if self.verbose:
            print(("      time (s): {:>10.2f}s").format(
                time.perf_counter()-time_init))

        # status date
        time_init = time.perf_counter()
        if self.verbose:
            print("   STATUS_DATE detection....")
        date_candidates = []
        for key in self._dict_column:
            if (
                    key not in cols_with_status and
                    first_approach_distincs[0][key] > 2):
                if (
                        self._dict_column[key]["dtype"] == "float" and
                        self._dict_column[key]["run"]["integer_float"]):
                    date_candidates.append(key)
                elif self._dict_column[key]["dtype"] != "float":
                    date_candidates.append(key)
        date_candidates_int = []
        date_candidates_str = []
        for elt in date_candidates:
            if self._dict_column[elt]["dtype"] in ["float", "integer"]:
                if (
                        len(str(int(self._dict_column[elt][
                            "run"]["first_value"]))) in [6, 8] and
                        is_date_integer([
                            int(self._dict_column[elt]["run"]["first_value"])
                            ])):
                    date_candidates_int.append(elt)
            else:
                if (
                        len(str(
                            self._dict_column[elt]["run"]["first_value"])) in
                        [7, 8, 9, 10] and is_date_object([
                            self._dict_column[elt]["run"]["first_value"]]
                        )):
                    date_candidates_str.append(elt)

        dates_check_str = date_test_object(
            self.df, date_candidates_str, self.special_values)
        dates_check_int = date_test_int(
            self.df, date_candidates_int, self.special_values)

        for elt in date_candidates_str:
            samples = (
                self._n_samples -
                self._dict_column[elt]["run"]["num_nans"] -
                self._dict_column[elt]["run"]["num_specials"])
            if samples == dates_check_str[0][elt]:
                self._dict_column[elt]["status"] = STATUS_DATE
        for elt in date_candidates_int:
            samples = (
                self._n_samples -
                self._dict_column[elt]["run"]["num_nans"] -
                self._dict_column[elt]["run"]["num_specials"])
            if samples == dates_check_int[0][elt]:
                self._dict_column[elt]["status"] = STATUS_DATE

        cols_with_status = _update_status(self._dict_column)
        if self.verbose:
            print(("      time (s): {:>10.2f}s").format(
                time.perf_counter()-time_init))

        # status numeric conversion
        time_init = time.perf_counter()
        if self.verbose:
            print("   STATUS_NUMERIC_CONVERSION detection....")
        num_candidates = []
        for key in self._dict_column:
            if (key not in cols_with_status):
                if self._dict_column[key]["dtype"] == "object":
                    if is_numeric_conversion(
                            [self._dict_column[key]["run"]["first_value"]]):
                        num_candidates.append(key)
        num_check = numeric_conv_test(
            self.df, num_candidates, self.special_values)
        for elt in num_candidates:
            samples = (
                self._n_samples -
                self._dict_column[elt]["run"]["num_nans"] -
                self._dict_column[elt]["run"]["num_specials"])
            if samples == num_check[0][elt]:
                self._dict_column[elt]["status"] = STATUS_NUMERIC_CONVERSION
        cols_with_status = _update_status(self._dict_column)
        if self.verbose:
            print(("      time (s): {:>10.2f}s").format(
                time.perf_counter()-time_init))

        # status ok
        for key in self._dict_column:
            if key not in cols_with_status:
                self._dict_column[key]["status"] = STATUS_OK

    def _find_duplicates(self):
        """Find duplicates."""

        time_init = time.perf_counter()
        if self.verbose:
            print("   Duplicates detection....")

        dupl_candidates = [key for key, info in self._dict_column.items()
                           if info["status"] in STATUS_VALID]
        dupl_first_test = find_duplicates_spark_rdd(
            self.df, dupl_candidates, self._dict_column,
            self._n_samples, 200)
        for key in dupl_first_test:
            if dupl_first_test[key]["count_duplicate"] == 1:
                self._dict_column[key]["duplicate"] = True
                self._dict_column[key][
                    "duplicate_of"] = dupl_first_test[key]["duplicated_of"]
        if self.verbose:
            print(("      time (s): {:>10.2f}s").format(
                time.perf_counter()-time_init))

    def _find_information_blocks(self):
        """Find information blocks"""

        time_init = time.perf_counter()
        if self.verbose:
            print("   Informational blocks detection....")

        # find information blocks
        info_column = [(key, self._dict_column[key]["run"]["sum_block"])
                       for key in self._dict_column]

        # generate blocks of candidates to be grouped
        keys = set(list(zip(*info_column))[1])
        groups = [[n for (n, v) in info_column if v == key] for key in keys]

        # sort blocks by size
        info_blocks = sorted(groups, key=len, reverse=True)
        n_blocks = len(info_blocks)
        largest_block = len(info_blocks[0])
        ungrouped = sum(1 for b in info_blocks if len(b) == 1)
        self._info_blocks_stats = [
            n_blocks-ungrouped, largest_block, ungrouped]

        # information block id for each column
        for id_block, block_columns in enumerate(info_blocks):
            if len(block_columns) > 1:
                for column in block_columns:
                    self._dict_column[column]["block_id"] = id_block
        if self.verbose:
            print(("      time (s): {:>10.2f}s").format(
                time.perf_counter()-time_init))

    def _run(self):
        """Run all preprocessing steps and the decision engine."""
        time_init = time.perf_counter()

        self._n_samples = self.df.count()
        self._n_columns = len(self.df.columns)
        self._column_names = self.df.columns

        # find and tag occurrences
        self._find_cases()

        # find information blocks
        self._find_information_blocks()

        self._time_cases = (time.perf_counter() - time_init)
        time_init = time.perf_counter()

        # find duplicates
        self._find_duplicates()
        self._time_duplicates = (time.perf_counter() - time_init)
        time_init = time.perf_counter()

        # set actions
        self._decision_engine()

    def _transform_nan_unique(self):
        """Transform nan values for columns tagged as nan_unique.
        """
        if self.verbose:
            print("transforming nan unique....")
        maxlength = 8
        msg = "NaN values replaced by {}"
        dict_fill_na = {}
        for column in self._column_names:
            if self._dict_column[column]["status"] == "nan_unique":
                value = self._dict_column[column]["run"]["first_value"]
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
                dict_fill_na[column] = replace
                if self.verbose:
                    print("   " + column + ": " + msg.format(replace))
                # add comment
                self._dict_column[column]["comment"] = msg.format(replace)
        # replace NaN values, in-place operation
        self.df = self.df.fillna(dict_fill_na)

    def _transform_numeric_conversion(self):
        """Transform strings to float type for columns tagged as
        numeric_conversion"""
        if self.verbose:
            print("transforming numeric conversion....")
        for column in self._column_names:
            if self._dict_column[column]["status"] == "numeric_conversion":
                self.df = self.df.withColumn(
                    column, self.df[column].cast("float"))
                if self.verbose:
                    print("   Column {} tranformed to float".format(column))

    def _transform_remove(self, mode="basic"):
        """Remove columns tagged as 'remove'. If mode=aggressive, also columns
        tagged as 'review' are dropped."""
        if self.verbose:
            print("removing columns....")
        remove = [col for col in self._column_names
                  if self._dict_column[col]["action"] == "remove"]
        if mode is "aggressive":
            for column in self._column_names:
                in_rmv = (column not in remove)
                if self._dict_column[column]["status"] == "date":
                    if self.date is None or self.date != column and in_rmv:
                        remove.append(column)
                elif self._dict_column[column]["status"] == "id" and in_rmv:
                    remove.append(column)
        elif mode is "basic":
            if self.verbose:
                print("   only columns tagged as 'remove' are dropped.")
        self._n_columns_remove = len(remove)

        if self.verbose:
            print("   {} columns removed".format(self._n_columns_remove))
        # drop column in-place
        self.df = self.df.drop(*remove)
