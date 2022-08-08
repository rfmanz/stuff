"""
The :mod:`grmlab.modelling.feature_selection` module includes feature selection
optimization algorithm and auxiliary functions to calculate transformation
metrics.
"""

from .crfe import CRFE
from .optimize import SelectOptimize

from .util import condition_number
from .util import corrcoef_X_y
from .util import correlation_analysis


__all__ = ['CRFE',
           'SelectOptimize',
           'condition_number',
           'corrcoef_X_y',
           'correlation_analysis']