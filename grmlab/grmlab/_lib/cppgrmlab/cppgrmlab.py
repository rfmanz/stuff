"""
Wrappers for the CPPGRMLAB C++ library.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2018.

import os.path
import platform

import numpy as np
import numpy.ctypeslib as npct

from ctypes import byref, c_bool, c_double, c_int, POINTER

from .util import d_arr_1d, i_arr_1d, grmlab_int_type


# load library
libabspath = os.path.dirname(os.path.abspath(__file__))
system_os = platform.system()
linux_os = (system_os == "Linux" or "CYGWIN" in system_os)

if linux_os:
    cppgrmlab = npct.load_library("_cppgrmlab.so", libabspath)
else:
    cppgrmlab = npct.load_library("cppgrmlab.dll", libabspath)


# sorting, searching and counting functions
cppgrmlab.cppgrmlab_radixsort.restype = None
cppgrmlab.cppgrmlab_radixsort.argtypes = [i_arr_1d, c_int, c_int, c_int]


def radixsort(v, inplace=False):
    """
    Radix sort algorithm.

    TypeError will be raised when the following arguments are not instances of
    numpy.ndarray:
        v
    """
    from .util import grmlab_int_type_check_and_cast
    _v = grmlab_int_type_check_and_cast(v, 'v')
    if inplace:
        cppgrmlab.cppgrmlab_radixsort(_v, 0, len(v), 24)
        return _v
    else:
        _v_copy = _v.copy()  # default in-place C++ function -> return copy
        cppgrmlab.cppgrmlab_radixsort(_v_copy, 0, len(v), 24)
        return _v_copy


cppgrmlab.cppgrmlab_argsort_d.restype = None
cppgrmlab.cppgrmlab_argsort_d.argtypes = [c_int, d_arr_1d, i_arr_1d]


def argsort_d(v):
    """
    Arg sort algorithm. Significantly faster for large arrays with small amount
    of short numbers, about (1.25 - 1.4 speedup).

    TyperError will be raised when the following arguments are not instances of
    numpy.ndarray
        v
    """
    from .util import grmlab_double_type_check_and_cast
    _v = grmlab_double_type_check_and_cast(v, 'v')
    n = len(v)
    _idx = np.zeros(n, dtype=grmlab_int_type)
    cppgrmlab.cppgrmlab_argsort_d(n, _v, _idx)
    return _idx


cppgrmlab.cppgrmlab_argsort_i.restype = None
cppgrmlab.cppgrmlab_argsort_i.argtypes = [c_int, i_arr_1d, i_arr_1d]


def argsort_i(v):
    """
    Arg sort algorithm. Significantly faster for large arrays with small amount
    of short numbers, about (1.25 - 1.4 speedup).

    TyperError will be raised when the following arguments are not instances of
    numpy.ndarray
        v
    """
    from .util import grmlab_int_type_check_and_cast
    _v = grmlab_int_type_check_and_cast(v, 'v')
    n = len(v)
    _idx = np.zeros(n, dtype=grmlab_int_type)
    cppgrmlab.cppgrmlab_argsort_i(n, _v, _idx)
    return _idx


cppgrmlab.cppgrmlab_unique_d.restype = None
cppgrmlab.cppgrmlab_unique_d.argtypes = [c_int, d_arr_1d, d_arr_1d,
                                         POINTER(c_int), c_bool]


def unique_d(v, sort=True):
    """
    Find unique values in double array. Significantly faster than numpy.unique
    when the number of distinct elements is less than ~10% of total length.

    TyperError will be raised when the following arguments are not instances of
    numpy.ndarray
        v
    """
    from .util import grmlab_double_type_check_and_cast
    _v = grmlab_double_type_check_and_cast(v, 'v')
    _u = np.empty_like(_v)
    _size = c_int()
    cppgrmlab.cppgrmlab_unique_d(len(_v), _v, _u, byref(_size), sort)
    return _u[:_size.value]


cppgrmlab.cppgrmlab_unique_i.restype = None
cppgrmlab.cppgrmlab_unique_i.argtypes = [c_int, i_arr_1d, i_arr_1d,
                                         POINTER(c_int), c_bool]


def unique_i(v, sort=True):
    """
    Find unique values in integer array. Significantly faster than numpy.unique
    when the number of distinct elements is less than ~50% of total length.
    It uses internally the radixsort algorithm.

    TyperError will be raised when the following arguments are not instances of
    numpy.ndarray
        v
    """
    from .util import grmlab_int_type_check_and_cast
    _v = grmlab_int_type_check_and_cast(v, 'v')
    _u = np.empty_like(_v)
    _size = c_int()
    cppgrmlab.cppgrmlab_unique_i(len(_v), _v, _u, byref(_size), sort)
    return _u[:_size.value]


cppgrmlab.cppgrmlab_unique_i_count.restype = None
cppgrmlab.cppgrmlab_unique_i_count.argtypes = [c_int, i_arr_1d, i_arr_1d,
                                               i_arr_1d, POINTER(c_int)]


def unique_i_count(v):
    """
    Equivalent to np.unique(v, return_counts=True)
    """
    from .util import grmlab_int_type_check_and_cast
    _v = grmlab_int_type_check_and_cast(v, 'v')
    _u = np.empty_like(_v)
    _c = np.empty_like(_v)
    _size = c_int()
    cppgrmlab.cppgrmlab_unique_i_count(len(_v), _v, _u, _c, byref(_size))
    return _u[:_size.value], _c[:_size.value]


cppgrmlab.cppgrmlab_pearsonr_d_i.restype = c_double
cppgrmlab.cppgrmlab_pearsonr_d_i.argtypes = [c_int, d_arr_1d, i_arr_1d]


def pearsonr_d_i(x, y):
    """
    Pearson r correlation coefficient with integral target.

    TypeError will be raised when the following arguments are not instances of
    numpy.ndarray:
        x, y
    """
    from .util import grmlab_int_type_check_and_cast
    from .util import grmlab_double_type_check_and_cast
    _x = grmlab_double_type_check_and_cast(x, 'x')
    _y = grmlab_int_type_check_and_cast(y, 'y')
    return cppgrmlab.cppgrmlab_pearsonr_d_i(len(x), _x, _y)


cppgrmlab.cppgrmlab_pearsonr_i_i.restype = c_double
cppgrmlab.cppgrmlab_pearsonr_i_i.argtypes = [c_int, i_arr_1d, i_arr_1d]


def pearsonr_i_i(x, y):
    """
    Pearson r correlation coefficient with integral target.

    TypeError will be raised when the following arguments are not instances of
    numpy.ndarray:
        x, y
    """
    from .util import grmlab_int_type_check_and_cast
    _x = grmlab_int_type_check_and_cast(x, 'x')
    _y = grmlab_int_type_check_and_cast(y, 'y')
    return cppgrmlab.cppgrmlab_pearsonr_i_i(len(x), _x, _y)


cppgrmlab.cppgrmlab_pearsonr_d_d.restype = c_double
cppgrmlab.cppgrmlab_pearsonr_d_d.argtypes = [c_int, d_arr_1d, d_arr_1d]


def pearsonr_d_d(x, y):
    """
    Pearson r correlation coefficient with double target.

    TypeError will be raised when the following arguments are not instances of
    numpy.ndarray:
        x, y
    """
    from .util import grmlab_double_type_check_and_cast
    _x = grmlab_double_type_check_and_cast(x, 'x')
    _y = grmlab_double_type_check_and_cast(y, 'y')
    return cppgrmlab.cppgrmlab_pearsonr_d_d(len(x), _x, _y)


cppgrmlab.cppgrmlab_pearsonr_ctree_sort_b.restype = c_double
cppgrmlab.cppgrmlab_pearsonr_ctree_sort_b.argtypes = [c_int, i_arr_1d, c_int,
                                                      c_int]


def pearsonr_ctree_sort_b(y, m, t):
    """
    Pearson r correlation coefficient for CTree implementation with binary
    target.

    TypeError will be raised when the following arguments are not instances of
    numpy.ndarray:
        y
    """
    from .util import grmlab_int_type_check_and_cast
    _y = grmlab_int_type_check_and_cast(y, 'y')
    return cppgrmlab.cppgrmlab_pearsonr_ctree_sort_b(len(y), _y, m, t)


cppgrmlab.cppgrmlab_pearsonr_ctree_sort_d.restype = c_double
cppgrmlab.cppgrmlab_pearsonr_ctree_sort_d.argtypes = [c_int, d_arr_1d, c_int]


def pearsonr_ctree_sort_d(y, m):
    """
    Pearson r correlation coefficient for CTree implementation with double
    target.

    TypeError will be raised when the following arguments are not instances of
    numpy.ndarray:
        y
    """
    from .util import grmlab_double_type_check_and_cast
    _y = grmlab_double_type_check_and_cast(y, 'y')
    return cppgrmlab.cppgrmlab_pearsonr_ctree_sort_d(len(y), _y, m)
