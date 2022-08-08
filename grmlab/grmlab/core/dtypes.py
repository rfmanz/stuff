"""
GRMlab functions for checking data types.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import numbers
import re

import numpy as np


def is_numpy_int(dtype):
    """
    Check whether a numpy type is of the integer type.

    Parameters
    ----------
    dtype: numpy.dtype

    Example
    -------
    >>> import numpy as np
    >>> fron grmlab.core.dtypes import is_numpy_int
    >>> a = np.ones(10)
    >>> is_numpy_float(a.dtype)
    False
    >>> b = np.ones(10, dtype=np.int)
    >>> is_numpy_float(b.dtype)
    True
    """
    kind = dtype.kind
    if kind in np.typecodes["Integer"]:
        return True
    return False


def is_numpy_float(dtype):
    """
    Check whether a numpy type is of the float type.

    Parameters
    ----------
    dtype: numpy.dtype

    Example
    -------
    >>> import numpy as np
    >>> fron grmlab.core.dtypes import is_numpy_float
    >>> a = np.ones(10)
    >>> is_numpy_float(a.dtype)
    True
    >>> b = np.ones(10, dtype=np.int)
    >>> is_numpy_float(b.dtype)
    False
    """
    kind = dtype.kind
    if kind in np.typecodes["Float"]:
        return True
    return False


def is_numpy_object(dtype):
    """
    Check whether a numpy type is of the object or character type.

    Parameters
    ----------
    dtype: numpy.dtype

    Example
    -------
    >>> import numpy as np
    >>> from grmlab.core.dtypes import is_numpy_object
    >>> c = np.array(["a", "b", "Dcfasdfd12"])
    >>> is_numpy_object(c.dtype)
    True
    """
    kind = dtype.kind
    if kind in np.typecodes["Character"] or kind in ['O', 'U']:
        return True
    return False


def is_binary(x):
    """
    Check whether an array-like contains only binary data.

    Parameters
    ----------
    x : array-like
    """
    u = np.unique(x)  # fastest for general purposes
    binary_01 = (len(u) == 2 and u[0] == 0 and u[1] == 1)
    binary_0 = all(x == 0)
    binary_1 = all(x == 1)
    return binary_01 or binary_0 or binary_1


def is_number(obj):
    """
    Check whether an object is of number type (int, long, float, complex).

    Parameters
    ----------
    obj : object
    """
    return isinstance(obj, numbers.Number)


def is_string(obj):
    """
    Check whether an object is of string type.

    Parameters
    ----------
    obj : object
    """
    return isinstance(obj, str)


def check_dtype(name, dtype, variables_nominal, verbose=False):
    """
    Validate numpy data type and return data type string ("numerical",
    "ordinal", "categorical", "nominal").

    Check inconsistencies when variable if tagged as nominal, but it contains
    strings, therefore it should be treated as categorical instead.

    Parameters
    ----------
    name: str
        The variable name.

    dtype: numpy.dtype

    variables_nominal: array-like
        List of variable names.

    verbose : int or boolean (default=False)
        Controls verbosity of output.

    Example
    -------
    >>> import numpy as np
    >>> from grmlab.core.dtypes import check_dtype
    >>> variable = "A"
    >>> variables_nominal = ["A", "B"]
    >>> a = np.asarray(["1", "2", "3"])
    >>> check_dtype(variable, a.dtype, variables_nominal, True)
    datatype-check: variable A tagged as nominal shall be treated as
    categorical. 'categorical'
    """
    if not isinstance(dtype, np.dtype):
        raise TypeError("dtype is not a numpy.dtype object.")

    if is_numpy_int(dtype):
        var_dtype = "ordinal"
    elif is_numpy_float(dtype):
        var_dtype = "numerical"
    elif is_numpy_object(dtype):
        var_dtype = "categorical"

    if name in variables_nominal:
        if is_numpy_object(dtype):
            if verbose:
                print("datatype-check: variable {} tagged as nominal "
                      "shall be treated as categorical.".format(name))
            var_dtype = "categorical"
        else:
            var_dtype = "nominal"

    return var_dtype


def check_target_dtype(target):
    """
    Validate target data and return target type string ("binary",
    "multi-class integer", "numerical", "multi-class categorical")

    Parameters
    ----------
    target : numpy array.

    Examples
    --------
    >>> import numpy as np
    >>> from grmlab.core.dtypes import check_target_dtype
    >>> a = np.asarray([0, 1, 0, 1])
    >>> check_target_dtype(a)
    'binary'
    >>> b = np.asarray([0.1, 0.9, 0.5, 0.2])
    >>> check_target_dtype(b)
    'numerical'
    >>> c = np.asarray([1, 3, 2, 1])
    >>> check_target_dtype(c)
    'multi-class integer'
    >>> d = np.asarray(["A", "B", "C", "D"])
    >>< check_target_dtype(d)
    'multi-class categorical'
    """
    if not isinstance(target, np.ndarray):
        raise TypeError("target is not a numpy.ndarray object.")

    if is_numpy_object(target.dtype):
        return "multi-class categorical"
    elif is_numpy_int(target.dtype):
        if is_binary(target):
            return "binary"
        else:
            return "multi-class integer"
    elif is_numpy_float(target.dtype):
        return "numerical"


def check_date_format(dates):
    """
    Validate dates data for GRMlab processes. Date format must be an
    integer yyyymm.

    Parameters
    ----------
    dates : numpy array

    Examples
    --------
    >>> import numpy as np
    >>> from grmlab.core.dtypes import check_date_format
    >>> dates = np.array([20120102])
    >>> check_date_format(dates)
    ValueError: date variable has incorrect format. Correct format: integer
    yyyymm.
    >>> dates = np.array([201201])
    >>> check_date_format(dates)
    """
    if not isinstance(dates, np.ndarray):
        raise TypeError("dates is not a numpy.ndarray object.")

    if not is_numpy_int(dates.dtype):
        raise TypeError("dates must be of type integer")

    regex = re.compile("[12]\d{3}(0[1-9]|1[0-2])")
    vectorize_regex = np.vectorize(lambda x: len(str(x)) == 6 and
                                   bool(regex.match(str(x))))

    if any(vectorize_regex(dates) == 0):
        raise ValueError("date variable has incorrect format. "
                         "Correct format: integer yyyymm.")
