"""
The :mod:`grmlab.extensions.multivariate` module implements
multivariate analysis
"""

from .base import MultivariateAnalysis
from .correlation import MultivariateCorrelations

__all__ = ['MultivariateAnalysis',
		   'MultivariateCorrelations']