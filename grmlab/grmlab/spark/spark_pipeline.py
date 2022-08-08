"""
GRMlab-Spark pipeline.
"""

# Authors: Fernando Gallego Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2019.

import time

from .core.base_spark import GRMlabSparkBase
from .data_processing.preprocessing.spark_preprocessing import PreprocessingSpark
from .data_analysis.spark_univariate import UnivariateSpark


class SparkPipeline(GRMlabSparkBase):
    """
    Top layer class for all grmlab modules in Spark.

    .. note::

        There are three types of parameters:
            #. Parameters to define the Spark session: name, n_cores, \
               executor_memory, executor_cores, and memory overhead.
            #. Parameters to load and save the database: path_db, \ 
               path_db_result, fileFormat, and sep.
            #. Parameters common to all modules in the modelling process: \ 
               target, date, special_values, and variables_nominal.

    Parameters
    ----------
    path : str
        Path in HDFS where the file is located.

    fileFormat : str (default='csv')
        Only works with 'csv' or 'parquet' file formats.

    name : str or None (default="")
        The name of your application. This will appear in the UI and in
        log data.
        https://spark.apache.org/docs/latest/configuration.html

    spark : pyspark.sql.session.SparkSession
        Entry point to the services of Spark.

    result_path : str
        Path in HDFS to store the result after the transformation.

    sep : str (default=',')
        Column separator for csv files.

    target : str or None (default=None)
        The name of the column flagged as target.

    date : str or None (default=None)
        The name of the column flagged as date.

    variables_nominal : list or None (default=None)
        List of ordinal variables to be treated as nominal.

    special_values : list or None (default=None)
        List of special values to be considered.

    verbose : int or boolean (default=False)
        Controls verbosity of output.
    """
    def __init__(self, path, fileFormat, spark, result_path=None,
                 sep=None, target=None, date=None,
                 variables_nominal=[], special_values=[],
                 verbose=True):
        self.target = target
        self.date = date
        self.path = path
        self.fileFormat = fileFormat
        self.sep = sep
        self.result_path = result_path
        self.variables_nominal = variables_nominal
        self.special_values = special_values

        # stages variables
        self.preprocessing = None
        self.univariate = None
        self.bivariate = None

        # variables info
        self._dict_steps = {}

        # Spark parameters
        self.spark = spark

        self.verbose = verbose

        # times
        self._time_spark = 0
        self._time_load_file = 0

        # initialize Spark and load file
        time_init = time.perf_counter()

        if self.verbose:
            print("loading file....")

        self.load_file(self.path, self.fileFormat, self.sep)

        self._time_load_file = (time.perf_counter() - time_init)
        if self.verbose:
            print(
                ("file loaded - time (s): {:>10.2f}s").format(
                 self._time_load_file))

    def init(self, module, kill=False, *args, **kwargs):
        """
        Initializes any module present in grmlab/spark

        Parameters
        ----------
        module : str
            The current available modules are 'preprocessing' and 'univariate'.

        kill : boolean (default=False)
            If false and preprocessing is already defined it will produce this
            value error. Only with kill=True will overwrite the previous
            preprocessing class.

        *args : list
            Variable length argument list.

        **kwargs : keyword arguments
            Additional arguments for the initialization of the module. Check
            the classes :class:`PreprocessingSpark` or :class:`UnivariateSpark`
            for details.

            .. note::

                The arguments: target, date, special_values, variables_nominal
                and verbose are already loaded in the initialization of this
                class.

        """
        self._check_module(module)
        if module == "preprocessing":
            self._check_if_module_is_initialized(self.preprocessing, kill)
            self.preprocessing = PreprocessingSpark(
                df=self.df, target=self.target, date=self.date,
                special_values=self.special_values, verbose=self.verbose,
                *args, **kwargs)
        elif module == "univariate":
            self._check_if_module_is_initialized(self.univariate, kill)
            self.univariate = UnivariateSpark(
                df=self.df, spark_context=self.spark,
                target=self.target, date=self.date,
                special_values=self.special_values,
                variables_nominal=self.variables_nominal, verbose=self.verbose,
                *args, **kwargs)

    def run(self, module, *args, **kwargs):
        """
        Executes method :mod:`run` of any previously initialized module.

        Parameters
        ----------
        module : str
            The current available modules are 'preprocessing' and 'univariate'.

        *args : list
            Variable length argument list.

        **kwargs : keyword arguments
            Arguments for the run method of the module. Check the classes
            :class:`PreprocessingSpark` or :class:`UnivariateSpark` for
            details.
        """
        self._check_module(module)
        if module == "preprocessing":
            self.preprocessing.run(*args, **kwargs)
            self._dict_steps[module] = self.preprocessing._dict_column
        elif module == "univariate":
            self.univariate.run(*args, **kwargs)
            self._dict_steps[module] = self.univariate._dict_variables

    def transform(self, module, overwrite_df=True, *args, **kwargs):
        """
        Executes method :mod:`transform` of any previously run module.

        Parameters
        ----------
        module : str
            The current available modules are 'preprocessing' and 'univariate'.

        overwrite_df : boolean (default=True)
            If True, it overwrites the self.df with the output of the
            transformation.

        *args : list
            Variable length argument list.

        **kwargs : keyword arguments
            Arguments for the transform method of the module. Check the classes
            :class:`PreprocessingSpark` or :class:`UnivariateSpark` for
            details.
        """
        self._check_module(module)
        if module == "preprocessing":
            self.preprocessing.transform(*args, **kwargs)
            if overwrite_df:
                self.df = self.preprocessing.df
        elif module == "univariate":
            self.univariate.transform(*args, **kwargs)
            if overwrite_df:
                self.df = self.univariate.df

    def save_df(self):
        """Saves result as a parquet file to hdfs"""
        self.df.write.parquet(self.result_path)

    def _check_module(self, module_name):
        """Check if the module is available"""
        if module_name in ["preprocessing", "univariate"]:
            pass
        elif module_name == "bivariate":
            raise NotImplementedError(
                "module {} is not implemented yet. Only modules "
                "univariate and preprocessing are available "
                "at the moment.".format(module_name))
        else:
            raise ValueError("module {} does not exist. Only modules "
                             "univariate and preprocessing are "
                             "valid".format(module_name))

    def _check_if_module_is_initialized(self, module_class, kill):
        if (module_class is not None) and (not kill):
            raise ValueError("Class {} is already defined. "
                             "Set kill to True if you want to reset"
                             "the class.".format(type(module_class)))
