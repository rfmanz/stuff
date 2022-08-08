"""
Utility functions for use with wrappers for the CPPGRMLAB C++ library.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2018.

import numpy.ctypeslib as npct

from numpy import int32
from numpy import float as numpy_float
from platform import architecture


# set integer type based on the python architecture and operating system.
# MS Windows C long int size is 32 bits in x86/x64. Numpy inherits the default
# size of an integer from C compiler long int. On Linux (and UNIX-based OS) C
# long int size is 64 bits in x64.
# grmlab_int_type is only used to set the dimension of integer arrays,
# therefore we use np.int32, this might change in the future.
python_arch, python_system = architecture()

if "Windows" in python_system:
    grmlab_int_type = int32
# elif python_arch == '64bit':
#   grmlab_int_type = int64
else:
    grmlab_int_type = int32


# input numpy ctypes
i_arr_1d = npct.ndpointer(dtype=grmlab_int_type, ndim=1, flags='CONTIGUOUS')
d_arr_1d = npct.ndpointer(dtype=numpy_float, ndim=1, flags='CONTIGUOUS')


def msg_err_ndarray(argname):
    """
    Return error message for argument argname when not being an instane of
    numpy.ndarray.
    """
    return 'Argument ' + argname + ' must be an instance of numpy.ndarray.'


def msg_err_bad_argtype(argname, exp_dtype):
    """
    Return error message for argument argname when not being of type exp_dtype.
    """
    from numpy import dtype
    str_dtype = str(dtype(exp_dtype))
    return 'Argument ' + argname + ' must be of type ' + str_dtype + "."


def grmlab_int_type_check_and_cast(arg, argname):
    """
    Return arg cast to an integer of the correct size for cppgrmlab, or as
    None.
    Raises TypeError if arg is not an instance of numpy.ndarray.
    Raises TypeError if arg is not an integer of the correct size.
    """
    from numpy import ndarray
    if arg is None:
        return None
    if not isinstance(arg, ndarray):
        raise TypeError(msg_err_ndarray(argname))
    if arg.dtype != grmlab_int_type:
        try:
            return arg.astype(grmlab_int_type)
        except Exception as err:
            raise TypeError(msg_err_bad_argtype(argname, grmlab_int_type))
    return arg


def grmlab_double_type_check_and_cast(arg, argname):
    """
    Return arg cast with python-arch integer type, or as None.
    Raises TypeError if arg is not an instance of numpy.ndarray.
    Raises TypeError if arg is not a numpy float.
    """
    from numpy import ndarray
    if arg is None:
        return None
    if not isinstance(arg, ndarray):
        raise TypeError(msg_err_ndarray(argname))
    if arg.dtype != numpy_float:
        try:
            return arg.astype(numpy_float)
        except Exception as err:
            raise TypeError(msg_err_bad_argtype(argname, numpy_float))
    return arg
