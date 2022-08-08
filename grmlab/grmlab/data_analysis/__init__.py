"""
The :mod:`grmlab.data_analysis` module includes methods and algorithms to 
perform exploratory data analysis on a given dataset.
"""

from .bivariate import Bivariate
from .bivariate_large import BivariateByColumns
from .bivariate_continuous import BivariateContinuous

from .multivariate.base import MultivariateAnalysis
from .multivariate.correlation import MultivariateCorrelations

from .univariate import Univariate
from .univariate_large import UnivariateByColumns


__all__ = ['Bivariate',
		   'BivariateByColumns',
		   'BivariateContinuous',
		   'Univariate',
		   'UnivariateByColumns',
		   'MultivariateAnalysis',
		   'MultivariateCorrelations']