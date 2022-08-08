"""
The :mod:`grmlab.modelling.model_analysis` module implements an exaustive
analysis of the models developed in :mod:`grmlab.modelling`.
"""

from .analyzer import ModelAnalyzer
from .analyzer_continuous import ModelAnalyzerContinuous
from .analyzer_continuous import LinearAssumptionsTest
from .comparison import ModelComparison

__all__ = ['ModelAnalyzer',
		   'ModelAnalyzerContinuous',
		   'LinearAssumptionsTest',
		   'ModelComparison']