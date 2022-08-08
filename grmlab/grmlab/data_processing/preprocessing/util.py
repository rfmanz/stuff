"""
Data preprocessing utils.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import re

import numpy as np
import pandas as pd

from ..._lib.cppgrmlab.cppgrmlab import unique_i


def is_date_integer(data):
    """Check whether integer data represent a date format."""
    # minor dimensionality test to check the adequacy of date formats.
    if len(str(data[0])) not in [6, 8]:
        return False

    # construct regex of several cases
    # regex 1: case yyyymm
    regex1 = r"((19|20)\d\d)(0?[1-9]|1[012])$"
    # regex 2: cases yyyymmdd
    regex2 = r"((19|20)\d\d)(0?[1-9]|1[012])(0[1-9]|[12]\d|3[01])$"
    regex = re.compile("|".join([regex1, regex2]))

    # vectorized implementation of regex for numpy
    vectorize_regex = np.vectorize(lambda x: bool(regex.match(str(x))))
    if all(vectorize_regex(data) == 1):
        return True
    else:
        return False


def is_date_object(data):
    """Check whether object data represent a date format."""
    # minor dimensionality test to check the adequacy of date formats.
    date_len = len(str(data[0]))
    if date_len < 7 or date_len > 10:
        return False

    # construct regex of several cases
    # regex 1: cases (yy)yy*mm*(dd)
    regex1 = (r"(?:\d{4}|\d{2})[^0-9a-zA-Z]{0,1}(0?[1-9]|1[012])"
              "[^0-9a-zA-Z]{0,1}((0[1-9]|[12]\d|3[01])?)$")
    # regex 2: cases (dd)*MMM*(yy)yy
    regex2 = (r"((0[1-9]|[12]\d|3[01])?)[^0-9a-zA-Z]{0,1}[a-zA-Z]{3}"
              "[^0-9a-zA-Z]{0,1}(?:\d{4}|\d{2})$")
    # regex 3: cases (dd)*mm*(yy)yy
    regex3 = (r"((0[1-9]|[12]\d|3[01])?)[^0-9a-zA-Z]{0,1}(0?[1-9]|1[012])"
              "[^0-9a-zA-Z]{0,1}(?:\d{4}|\d{2})$")
    regex = re.compile("|".join([regex1, regex2, regex3]))

    # vectorized implementation of regex for numpy
    vectorize_regex = np.vectorize(lambda x: bool(regex.match(str(x))))
    if all(vectorize_regex(data) == 1):
        return True
    else:
        return False


def is_exclude_case(names):
    """Returns column names belonging to category exclude."""
    regex1 = "(dec)?(score)?_[0-9]{1,2}_.*"
    regex2 = "(salida)?(variable)?_(alfa)?num(erica)?_[0-9]{1,2}_[0-9]{1,2}"
    regex3 = "tob_[0-9]{1,2}_.*"
    regex = re.compile("|".join([regex1, regex2, regex3]))

    return [c for c in names if bool(regex.match(c))]


def is_identifier(data, n_samples, threshold=0.9):
    """Check if data is likely to be an identifier."""
    n_uniques = len(pd.Series(data).unique())
    ratio_unique = n_uniques / len(data)
    min_n_samples = 0.1 * n_samples
    n = len(data)

    if ratio_unique > threshold and n_uniques > 2 and n > min_n_samples:
        return True
    else:
        return False


def is_binary_integer(data):
    """Check if data is binary."""
    u_values = unique_i(data)
    if (len(u_values) == 2 and (u_values[0] == 0 and u_values[1] == 1)):
        return True
    else:
        return False


def is_numeric_conversion(data):
    """Check whether column requires a numeric conversion."""
    regex = re.compile(r"(\d)+(,\d{2})")

    if not regex.match(str(data[0])):
        return False

    vectorize_regex = np.vectorize(lambda x: bool(regex.match(str(x))))
    if all(vectorize_regex(data) == 1):
        return True
    else:
        return False
