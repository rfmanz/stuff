"""
The :mod:`grmlab.extensions.inference` module implements the
inference algorithm based on kgb model obtained in :mod:`grmlab.modelling`.
"""

from .inferencer import Inferencer
from .sampling_bias import stabilityIndex

__all__ = ['Inferencer', 'stabilityIndex']