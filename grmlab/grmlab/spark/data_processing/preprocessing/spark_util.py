"""
Data preprocessing utils for Spark.
"""

# Authors: Fernando Gallego Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2019.

import re

import numpy as np
import pandas as pd
import pyspark.sql.functions as F

from hashlib import sha1
from pyspark.sql import Row
from pyspark.sql.types import IntegerType

from ....data_processing.preprocessing.algorithms import connected_pairs


def is_sparkSQL_int(dtype):
    """Check if sparksql type is integer type"""
    if dtype in ["int", "integer", "IntegerType"]:
        return True
    return False


def is_sparkSQL_float(dtype):
    """Check if sparksql type is float type"""
    if dtype in ["DoubleType", "double"]:
        return True
    return False


def decimal_to_float_int(dtype):
    if "decimal" not in dtype:
        return None
    dtype_ = dtype.split("(")[1].replace(")", "")
    nums = [int(num) for num in dtype_.split(",")]
    if nums[1] != 0:
        return "float"
    elif nums[1] == 0:
        return "int"


def count_nans_specials_first(df_spark, cols, special_values):
    """Counts the number of nans, specials and select the first entry that is
    neither nan nor special.

    It uses the Spark function mapPartitions and reduce.
    """

    def pandas_count_nans(chunk, cols, special):
        """treats each partition separately"""
        import pandas as pd
        import numpy as np
        df_pandas = pd.DataFrame(list(chunk), columns=cols)
        nan_vals = df_pandas.isna()
        special_vals = df_pandas.isin(special)
        first_vec = {}
        sum_block = {}
        for elt in cols:
            sum_block[elt] = np.packbits(nan_vals[elt]).sum()
            l1 = df_pandas[elt][~nan_vals[elt] & ~special_vals[elt]].values
            first = l1[0] if len(l1) > 0 else None
            first_vec[elt] = first
        result = [nan_vals.sum(), special_vals.sum(), first_vec, sum_block]
        df_pandas = None
        nan_vals = None
        special_vals = None
        return [result]

    def sum_nans(part1, part2):
        """aggregate the results of each partition"""
        for elt in part1[0].index:
            part1[0][elt] += part2[0][elt]
            part1[1][elt] += part2[1][elt]
            part1[3][elt] += part2[3][elt]
            if part1[2][elt] is None:
                part1[2][elt] = part2[2][elt]
        return part1

    return df_spark.select(cols).rdd.mapPartitions(
        lambda part: pandas_count_nans(part, cols, special_values)
        ).reduce(lambda part1, part2: sum_nans(part1, part2))


def count_nans_specials_first_rdd(df_spark, cols, special_values):
    """Counts the number of nans, specials and select the first entry that is
    neither nan nor special.

    It uses the Spark function map and reduce.
    """
    def check_nans_specials(row, special):
        """Function for map

        It check if an element is nan or special and store the value if it is
        not any of those types.
        """
        nan_bool = []
        special_bool = []
        first = []
        suma = []
        for elt in row:
            if elt is None:
                nan_bool.append(1)
                suma.append(128)
                special_bool.append(0)
                first.append(None)
            else:
                nan_bool.append(0)
                suma.append(0)
                if elt in special:
                    special_bool.append(1)
                    first.append(None)
                else:
                    special_bool.append(0)
                    first.append(elt)
        return [nan_bool, special_bool, first, suma, 1]

    def sum_nans(vec1, vec2):
        """Function for reduce

        it adds the number of nans and specials, stores one element neither nan
        nor special, and sum the boolean nans as an 8-bit number.
        """
        # value in decimal form of each character in an 8-bit number.
        vector_bin = [128, 64, 32, 16, 8, 4, 2, 1]
        for i in range(len(vec1[0])):
            # add nans and specials
            vec1[0][i] += vec2[0][i]
            vec1[1][i] += vec2[1][i]
            if vec1[2][i] is None:
                vec1[2][i] = vec2[2][i]
            # sum of packbits
            if vec2[0][i] == 1:
                vec1[3][i] += vector_bin[vec1[4]]
        if vec1[4] == 7:
            vec1[4] = 0
        else:
            vec1[4] += 1
        return vec1

    return df_spark.select(cols).rdd.map(
        lambda x: check_nans_specials(x, special_values)
        ).reduce(lambda vec1, vec2: sum_nans(vec1, vec2))


def count_distinct(df_spark, cols, rsd):
    """Counts the number of distinc values with the HLL approximation"""
    if len(cols) == 0:
        return [Row()]
    return df_spark.agg(*(
        F.approx_count_distinct(F.col(c), rsd=rsd).alias(c) for c in cols
        )).collect()


def count_constant(df_spark, cols, dict_values):
    """Counts the number of values equal to the first clean value"""
    if len(cols) == 0:
        return [Row()]
    return df_spark.select([F.when(
        F.col(c).isin([dict_values[c]["run"]["first_value"]]), 1
        ).otherwise(0).alias(c) for c in cols]
        ).select([F.sum(c).alias(c) for c in cols]).collect()


def integer_float_count(df_spark, cols):
    """Counts the number of float values equal to its integer part"""
    if len(cols) == 0:
        return [Row()]
    return df_spark.select([F.count(F.when(F.rint(x) == F.col(x), 1)
                                    ).alias(x) for x in cols]).collect()


def date_test_object(df_spark, cols, special):
    """Check whether object data represent a date format."""
    if len(cols) == 0:
        return [Row()]

    def is_date_object(s, special):
        """Check whether object data represent a date format."""
        if s in special:
            return 0
        # construct regex of several cases
        # regex 1: cases (yy)yy*mm*(dd)
        regex1 = r"(?:\d{4}|\d{2})[^0-9a-zA-Z]{0,1}(0?[1-9]|1[012])[^0-9a-zA-Z]{0,1}((0[1-9]|[12]\d|3[01])?)$"
        # regex 2: cases (dd)*MMM*(yy)yy
        regex2 = r"((0[1-9]|[12]\d|3[01])?)[^0-9a-zA-Z]{0,1}[a-zA-Z]{3}[^0-9a-zA-Z]{0,1}(?:\d{4}|\d{2})$"
        # regex 3: cases (dd)*mm*(yy)yy
        regex3 = r"((0[1-9]|[12]\d|3[01])?)[^0-9a-zA-Z]{0,1}(0?[1-9]|1[012])[^0-9a-zA-Z]{0,1}(?:\d{4}|\d{2})$"
        regex = re.compile("|".join([regex1, regex2, regex3]))
        # vectorized implementation of regex for numpy
        vectorize_regex = np.vectorize(lambda x: bool(regex.match(str(x))))
        if vectorize_regex(s) == 1:
            return 1
        else:
            return 0

    date_test_str = F.udf(lambda s: is_date_object(s, special), IntegerType())
    return df_spark.select(
        [F.count(
            F.when(date_test_str(c) == 1, 1).otherwise(None)
            ).alias(c) for c in cols]).collect()


def date_test_int(df_spark, cols, special):
    """Checks whether integer data represent a date format."""
    if len(cols) == 0:
        return [Row()]

    def is_date_integer(s, special):
        """Check whether integer data represent a date format."""
        if s in special:
            return 0
        # construct regex of several cases
        # regex 1: case yyyymm
        regex1 = r"((19|20)\d\d)(0?[1-9]|1[012])$"
        # regex 2: cases yyyymmdd
        regex2 = r"((19|20)\d\d)(0?[1-9]|1[012])(0[1-9]|[12]\d|3[01])$"
        regex = re.compile("|".join([regex1, regex2]))
        # vectorized implementation of regex for numpy
        vectorize_regex = np.vectorize(
            lambda x: bool(regex.match(str(int(x)))))
        if vectorize_regex(s) == 1:
            return 1
        else:
            return 0

    date_test_int = F.udf(lambda s: is_date_integer(s, special), IntegerType())
    return df_spark.select([
        F.count(
            F.when(date_test_int(c) == 1, 1).otherwise(None)
            ).alias(c) for c in cols
        ]).collect()


def numeric_conv_test(df_spark, cols, special):
    """Check whether column requires a numeric conversion."""
    if len(cols) == 0:
        return [Row()]

    def is_numeric_conversion(s, special):
        """Check whether column requires a numeric conversion."""
        # preguntar si ponerlo
        if s in special:
            return 0
        regex = re.compile(r"(\d)+(,\d{2})")
        vectorize_regex = np.vectorize(lambda x: bool(regex.match(str(x))))
        if vectorize_regex(s) == 1:
            return 1
        else:
            return 0

    num_test = F.udf(
        lambda s: is_numeric_conversion(s, special), IntegerType())
    return df_spark.select([F.count(
        F.when(num_test(c) == 1, 1).otherwise(None)).alias(c) for c in cols]
        ).collect()


def find_duplicates_spark_rdd(df_spark, cols, dict_column, n_samples,
                              n_sampling=100):

    duplicates_approx = df_spark.select(
        cols).sample(True, n_sampling/n_samples, 0.2565).collect()

    dtypes = [dict_column[elt]["dtype"] for elt in cols]
    grp_dtypes = pd.Index(cols).to_series().groupby(dtypes).groups

    # duplicates group candidates
    hash_value = {}
    for col in cols:
        array = np.array([vec[col] for vec in duplicates_approx])
        hash_value[col] = int(sha1(bytes("".join(
            map(str, array)).encode("utf8"))).hexdigest(), 16)

    # pair combinations
    combinations = []
    for type_key in grp_dtypes:
        names = grp_dtypes[type_key].values
        hash_type = np.array([hash_value[elt] for elt in names])
        idx = np.argsort(hash_type)
        hash_values = hash_type[idx]
        hash_names = names[idx]
        keys = np.unique(hash_values)
        occurrences = [hash_names[
            np.where(hash_values == key)] for key in keys]
        d = dict(zip(keys, occurrences))
        # form combinations
        names_to_compare = []
        for block in d.items():
            block_names = block[1]
            if len(block_names) > 1:
                names_to_compare.append(block_names)
        for i in range(len(names_to_compare)):
            bloque = list(names_to_compare[i])
            while(bloque):
                ref_elt = bloque.pop()
                for elt in bloque:
                    combinations.append((ref_elt, elt))

    def check_dup(row, comb):
        result = []
        row_dic = row.asDict()  # es fundamental para reducir tiempos
        for c1, c2 in comb:
            if row_dic[c1] != row_dic[c2]:
                result.append(1)
            else:
                result.append(0)
        return [result]

    def sum_dup(res1, res2):
        for i in range(len(res1[0])):
            res1[0][i] += res2[0][i]
        return res1
    result_dup = df_spark.rdd.map(
        lambda x: check_dup(x, combinations)
        ).reduce(lambda vec1, vec2: sum_dup(vec1, vec2))
    duplicated = [tuple(vec) for vec in np.array(combinations)[
                  np.array(result_dup[0]) == 0]]

    duplicate_columns = connected_pairs(duplicated)
    duplicates_sorted = [sorted(list(ls), key=len) for ls in duplicate_columns]
    duplicates_all = sum([l for l in duplicates_sorted], [])
    duplicated_vec = {}
    for column in cols:
        duplicated_vec[column] = {"duplicated_of": None, "count_duplicate": 0}
        if column in duplicates_all:
            column_block = next(block for block in duplicates_sorted if
                                column in block)
            parent = column_block[0]
            child = column_block[1:]
            if column in child:
                duplicated_vec[column]["duplicated_of"] = parent
            duplicated_vec[column]["count_duplicate"] = 1

    return duplicated_vec
