"""
The :mod:`grmlab.spark.data_analysis` module includes methods and algorithms to 
perform exploratory data analysis on a distributed data set in Spark.
"""

from .spark_univariate import UnivariateSpark


__all__ = ['UnivariateSpark']