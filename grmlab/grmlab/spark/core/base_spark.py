"""
GRMlab-SPARK base classes.
"""

# Authors: Fernando Gallego Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2019.

from pyspark.sql.session import SparkSession

from ...core.base import GRMlabBase


class GRMlabSparkBase(GRMlabBase):
    """
    Base class for all GRMlab SPARK classes.

    Parameters
    ----------
    spark : pyspark.sql.session.SparkSession
        Entry point to the services of Spark. SparkSession is now the new entry
        point of Spark that replaces the old SQLContext and HiveContext.
    """
    def __init__(self, spark):

        if not isinstance(spark, SparkSession):
            raise TypeError("spark must be "
                            "pyspark.sql.session.SparkSession.")
        self.spark = spark

        self.df = None
        self._n_partitions = None

    def load_file(self, path, fileFormat="csv", sep=",", header="true",
                  inferSchema="true", encoding="ISO-8859-1", **kwargs):
        """Loads a file within the Spark session

        Parameters
        ----------
        fileFormat : str (default='csv')
            only works with 'csv' or 'parquet' file formats.

        sep : str (default=',')
            column separator for csv files.
        """
        if fileFormat == "csv":
            self.df = self.spark.read.csv(
                path, sep=sep, header=header, inferSchema=inferSchema,
                encoding=encoding, **kwargs)
        elif fileFormat == "parquet":
            self.df = self.spark.read.option(
                "mergeSchema", "true").parquet(path)
        self._n_partitions = self.df.rdd.getNumPartitions()
