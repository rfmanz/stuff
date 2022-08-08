"""
Model Analyzer base
"""

# Authors: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2020.

import numbers
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_X_y

from ...core.base import GRMlabBase
from ...core.exceptions import NotFittedException
from ...core.exceptions import NotRunException
from ...core.dtypes import check_date_format
from ..base import GRMlabModel
from ..classification import GRMlabModelClassification
from ..regression import GRMlabModelRegression
from ..model_optimization import ModelOptimizer
from .plots import GaussianFilter
from .plots import plot_correlation_matrix


_METRIC_RAND_COMPUTED = []

_TEMPORAL_VARIABLES = []

_METRICS = {}

DICT_STRUCT = {'global_value': np.nan, 'name': None}

DICT_RANDOM = {'global_value': np.nan, 'name': None,
               'values_rand': None, 'std_rand': np.nan, 'skew_rand': np.nan,
               'kurtosis_rand': np.nan, 'mean_rand': np.nan,
               'median_rand': np.nan, 'p75_rand': np.nan, 'p25_rand': np.nan,
               'p95_rand': np.nan, 'p5_rand': np.nan, 'p99_rand': np.nan,
               'p1_rand': np.nan}


def basic_statistics(values):
    """
    Calculates distribution parameters of the input data.

    The distribution metrics are::

        metrics = ['mean', std', 'skew', 'excess kurtosis', 'media',
                   'P-1', 'P-5', 'P-25', 'P-75', 'P-95', 'P-99']

    Parameters
    ----------
    values : list() of float

    Returns
    -------
    dict_rand : dict.
        Returns a dictionary where the "key" is the name of the metric and the
        "value" is the obtained value of the metric.
    """
    dict_rand = {}
    dict_rand['mean_rand'] = np.mean(values)
    dict_rand['std_rand'] = np.std(values)
    dict_rand['skew_rand'] = skew(values)
    dict_rand['kurtosis_rand'] = kurtosis(values)

    [p1, p5, p25, median, p75, p95, p99] = np.percentile(values, [1, 5, 25,
                                                         50, 75, 95, 99])

    dict_rand['p1_rand'] = p1
    dict_rand['p5_rand'] = p5
    dict_rand['p25_rand'] = p25
    dict_rand['median_rand'] = median
    dict_rand['p75_rand'] = p75
    dict_rand['p95_rand'] = p95
    dict_rand['p99_rand'] = p99

    return dict_rand


class ModelAnalyzerBase(GRMlabBase):
    """Analysis a fitted model with a given data.

    It presents the characteristics of the model, calculates metrics to
    evaluate the model predictive performance, and study the independent
    variables used to fit the model.

    Parameters
    ----------
    name : str
        Name given to the model analyzer.

    model : :class:`GRMlabModel`, :class:`GRMlabModelClassification` or
            :class:`ModelOptimizer`
        The fitted model to be analyzed.

    n_simulations : int or None (default=100)
        Indicates the number of simulations to be run. It must be
        positive. If None, no simulations will be done.

    simulation_size : float (default=0.5)
        Size of the simulation data relative to number of elements in X.
        The value must be in the interval (0, 1).

    seed : int or 1-d array_like, optional
        Seed for RandomState. Must be convertible to 32 bit unsigned integers.

    feature_analysis_approximation : boolean (default=False)
        If True, it calculates only first order approximations for
        non-analytical calculations.

    verbose : boolean (default=True)
        Controls verbosity of output.
    """
    def __init__(self, name, model, n_simulations=100, simulation_size=1.,
                 seed=None, feature_analysis_approximation=False,
                 verbose=True):

        # model variables
        self.name = name
        self.model = model
        self.n_simulations = n_simulations
        self.simulation_size = simulation_size
        self.seed = seed
        self.feature_analysis_approximation = feature_analysis_approximation
        self.verbose = verbose

        # data
        self._n_samples = None
        self._n_columns = None

        # model variables
        self._model = None
        self._optimizer = None
        self._model_instance = None

        # metrics
        self._dict_metrics = {}

        # temporal
        self._count_temporal = None
        self._intervals_temporal = None

        # model features analysis
        self._feature_names = []
        self._feature_coefficients = None
        self._feature_importance = None

        self._correlations = None
        self._correlation_metrics = []

        self._vif = {}

        # curves
        self._roc = None

        # timing
        self._time_run = None
        self._time_run_metrics = None
        self._time_run_feature_analysis = None

        # flags
        self._is_dates_provided = False
        self._is_run = False

    def run(self, X, y, dates=None):
        """
        Runs all the analysis. Metrics, simulations, time analysis,
        feature importance, feature correlations and features VIF value.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
                The test input samples.

        y : array-like, shape = [n_samples]
            The test target values.

        dates : array-like, shape = [n_samples] or [n_dates] (default=None)
            Dates on the test data. This parameter is needed to implement
            a time analysis with multiple ModelAnalyzer results.
        """
        # check X, y, and dates formats and shape
        X, y = check_X_y(X, y, "csc")
        self._n_samples, self._n_columns = X.shape

        if dates is not None:
            dates = check_X_y(X, dates, "csc")[1]
            check_date_format(dates)
            self._is_dates_provided = True

        # check model
        if isinstance(self.model, (GRMlabModelClassification, GRMlabModel,
                                   GRMlabModelRegression)):
            if not self.model._is_estimator_fitted:
                raise NotFittedException(self.model)
            else:
                self._model = self.model
        elif isinstance(self.model, ModelOptimizer):
            self._model_instance = "optimizer"
            self._optimizer = self.model
            self._model = self.model.best_model
        else:
            raise TypeError("class {} is not an accepted model class.".
                            format(type(self.model).__name__))

        # check n_simulations and simulation_size
        if self.n_simulations is not None:
            if (not isinstance(self.n_simulations, (int, np.int)) or
                    self.n_simulations <= 0):
                raise ValueError("n_simulations must be \
                                  a positive integer or None")

            if isinstance(self.simulation_size, float):
                if not (self.simulation_size > 0):
                    raise ValueError("simulation_size must be positive.")
            else:
                raise TypeError("simulation_size must be float.")

        # Definition of the dictionary to store the results of the metrics.
        self._init_dict()

        self._run(X, y, dates)

        self._is_run = True

    def metric_plot_descriptive(self, name, plot_type="hist", hist_sigma=2):
        """Descriptive plots of the simulated metric.

        There are two plots available: histogram of the metric with its kernel
        and scatter plot with all the simulation results of the metric.
        Only the metrics with simulations are allowed::

            metrics = ['gini', 'kappa', 'mcc']

        Parameters
        ----------
        name : str
            Shortname of the metric.

        plot_type : str (default="hist")
            Two options: "hist" for the histogram, and "scatter" for the
            scattered plot.

        hist_sigma : int or float (default=2)
            Parameter of the gaussian kernel. The lower the sigma the higher
            the over-fitting. It must be positive.
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        if plot_type not in ("hist", "scatter"):
            raise ValueError("plot_type {} not supported.".format(plot_type))

        self._check_rand_metric(name)

        if (self.n_simulations is None) or (self.n_simulations <= 1):
            raise ValueError("n_simulations is < 2 or None, nothing to plot.")

        if plot_type == "hist":
            if not isinstance(hist_sigma, numbers.Number) or hist_sigma < 0:
                raise ValueError("hist_sigma must be a positive number.")

            GsF = GaussianFilter(data=self._dict_metrics[name]['values_rand'],
                                 sigma=hist_sigma)
            GsF.plot(self._dict_metrics[name]['name'])

        elif plot_type == "scatter":
            plt.plot(self._dict_metrics[name]['values_rand'], 'bo', alpha=0.6)
            plt.plot([self._dict_metrics[name]['global_value']] *
                     self.n_simulations, 'r-')
            plt.plot([self._dict_metrics[name]['p75_rand']] *
                     self.n_simulations, 'r--')
            plt.plot([self._dict_metrics[name]['p25_rand']] *
                     self.n_simulations, 'r--')
            plt.plot([self._dict_metrics[name]['p95_rand']] *
                     self.n_simulations, 'k--')
            plt.plot([self._dict_metrics[name]['p5_rand']] *
                     self.n_simulations, 'k--')
            plt.plot([self._dict_metrics[name]['p99_rand']] *
                     self.n_simulations, 'k-.')
            plt.plot([self._dict_metrics[name]['p1_rand']] *
                     self.n_simulations, 'k-.')

            range_val = (max(self._dict_metrics[name]['values_rand']) -
                         min(self._dict_metrics[name]['values_rand']))

            plt.text(-1, self._dict_metrics[name]['p75_rand'] +
                     range_val * 0.01, "P75", color="red")
            plt.text(-1, self._dict_metrics[name]['p95_rand'] +
                     range_val * 0.01, "P95", color="black")
            plt.text(-1, self._dict_metrics[name]['p99_rand'] +
                     range_val * 0.01, "P99", color="black")
            plt.text(-1, self._dict_metrics[name]['p1_rand'] +
                     range_val * 0.01, "P01", color="black")
            plt.text(-1, self._dict_metrics[name]['p5_rand'] +
                     range_val * 0.01, "P05", color="black")
            plt.text(-1, self._dict_metrics[name]['p25_rand'] +
                     range_val * 0.01, "P25", color="red")
            plt.text(-1, self._dict_metrics[name]['global_value'] +
                     range_val * 0.01, "Main", color="red")

            plt.xlabel("simulations")
            plt.ylabel(self._dict_metrics[name]['name'])
            plt.title("{} Simulation Results".format(
                self._dict_metrics[name]['name']))
            plt.show()
            plt.close()

    def metric_plot_temporal(self, name, display_obs=False):
        """Plots temporal analysis for a metric.

        The available metrics for temporal analysis are::

            metrics_binary = ['gini', 'default_rate', 'balanced_accuracy',
                'balanced_error_rate', 'diagnostic_odds_ratio',
                'discriminant_power', 'fnr', 'fpr', 'geometric_mean',
                'positive_likelihood', 'tpr','tnr', 'youden']

            metrics_continuous = ['r2_score', "mean_absolute_error", "mean_squared_error",
                "root_mean_squared_error", "pearson_corr",
                "spearson_rank_corr", "kendall_rank_corr"]

        Parameters
        ----------
        name : str
            name of the metric to be plotted.

        display_obs : boolean (default=False)
            If True, it shows the population of each time period.
        """
        _BLUE_COLOR = "#66CDAA"
        _GREEN_COLOR = "#2E8B57"

        if not self._is_run:
            raise NotRunException(self, "run")

        if not self._is_dates_provided:
            raise ValueError("dates is not provided. Temporal analysis was "
                             "not run.")

        self._check_temp_metric(name)

        def formatter_semester(x, pos=None):
            return x_axis[x] if x % 12 in (6, 12) else ""
        mt = mticker.FuncFormatter(formatter_semester)

        x_axis = [str(month[0]) for month in
                  self._intervals_temporal]

        y_axis = self._dict_metrics[name]['temporal']
        y_axis_no_nan = [elt for elt in self._dict_metrics[name]['temporal']
                         if elt is not None]

        fig, ax = plt.subplots()
        ax.plot(x_axis, y_axis, color=_GREEN_COLOR)
        plt.title("{} time analysis".format(
            self._dict_metrics[name]['name']))
        ax.set_xlabel('Time')
        ax.set_ylabel('1-year {}'.format(self._dict_metrics[name]['name']))
        ax.xaxis.set_major_formatter(mt)

        if display_obs:
            y_second_axis = [elt if (elt is not None) else np.nan for elt
                             in self._count_temporal]
            second_ax = ax.twinx()
            second_ax.xaxis.set_major_formatter(mt)
            second_ax.bar(x_axis, y_second_axis, alpha=0.5, color=_BLUE_COLOR)

        for lab in ax.get_xticklabels():
            lab.set_rotation(30)

        std_metric = np.std(y_axis_no_nan)

        report = (
            "\033[94m================================================================================\033[0m\n"
            "\033[1m\033[94m                    GRMlab Temporal Model Analysis                       \033[0m\n"
            "\033[94m================================================================================\033[0m\n"
            "\n"
            " \033[1mMetric: {:>10}\033[0m\n"
            "   Global value:              {:>0}\n"
            "   Range:              {:>0}-{:>0}\n"
            "   Standard deviation:        {:>0}\n"
            "   \033[94m--------------------------------------------------------------------------\033[0m\n"
            ).format(self._dict_metrics[name]['name'],
                     round(self._dict_metrics[name]['global_value'], 4),
                     round(np.min(y_axis_no_nan), 4),
                     round(np.max(y_axis_no_nan), 4),
                     round(std_metric, 4))
        print(report)

    def metric_stats(self, name):
        """Main statistics of the metrics distribution.

        Only the metrics with simulations are allowed::

            metrics_binary = ['gini', 'kappa', 'mcc']

            metrics_continuous = ['r2_score', "mean_absolute_error",
                                  "mean_squared_error",
                                  "root_mean_squared_error", "pearson_corr",
                                  "spearson_rank_corr", "kendall_rank_corr"]

        Parameters
        ----------
        name : str
            Shortname of the metric.
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        self._check_rand_metric(name)

        report = (
            "\033[94m====================================================\033[0m\n"
            "\033[1m\033[94m  GRMlab Model Analysis ({})                         \033[0m\n"
            "\033[94m====================================================\033[0m\n"
            "\n"
            "   Main value  {:>8.7f}\n"
            "\n"
            " \033[1mSimulation Results\033[0m\n"
            "   number of simulations  {:>8}\n"
            "   Mean                   {:>8.7f}\n"
            "   Standard Deviation     {:>8.7f}\n"
            "   Skewness               {:>8.7f}\n"
            "   Excess Kurtosis        {:>8.7f}\n"
            "   ---\n"
            "   P-99                   {:>8.7f}\n"
            "   P-95                   {:>8.7f}\n"
            "   P-75                   {:>8.7f}\n"
            "   Median                 {:>8.7f}\n"
            "   P-25                   {:>8.7f}\n"
            "   P-5                    {:>8.7f}\n"
            "   P-1                    {:>8.7f}\n"
            "   \033[94m------------------------------------------------\033[0m\n"
            ).format(self._dict_metrics[name]['name'],
                     self._dict_metrics[name]['global_value'],
                     len(self._dict_metrics[name]['values_rand']),
                     self._dict_metrics[name]['mean_rand'],
                     self._dict_metrics[name]['std_rand'],
                     self._dict_metrics[name]['skew_rand'],
                     self._dict_metrics[name]['kurtosis_rand'],
                     self._dict_metrics[name]['p99_rand'],
                     self._dict_metrics[name]['p95_rand'],
                     self._dict_metrics[name]['p75_rand'],
                     self._dict_metrics[name]['median_rand'],
                     self._dict_metrics[name]['p25_rand'],
                     self._dict_metrics[name]['p5_rand'],
                     self._dict_metrics[name]['p1_rand'])
        print(report)

    def model_characteristics(self):
        """The characteristics of the Model.

        Note that no analysis result is returned.
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        if self._model.feature_selection is None:

            report = (
                      "\033[94m================================================================================\033[0m\n"
                      "\033[1m\033[94m                         GRMlab Model Analysis                       \033[0m\n"
                      "\033[94m================================================================================\033[0m\n"
                      "\n"
                      " \033[1mGeneral information\033[0m\n"
                      "   model name                                  {:>30}\n"
                      "   number of records                           {:>30}\n"
                      "   number of variables                         {:>30}\n"
                      "   \033[94m--------------------------------------------------------------------------\033[0m\n"
                      "\n"
                      " \033[1mEstimator: Configuration Options\033[0m\n"
                      "   estimator class                             {:>30}\n" +
                      "".join(["   " + elt.replace("_", " ") +
                               (44 - len(elt)) * " " + "{:>30}\n" for elt in
                               self._model.get_params_estimator().keys()]) +
                      "   \033[94m--------------------------------------------------------------------------\033[0m\n"
                      ).format(self._model.name,
                               self._model._n_samples,
                               self._model.get_support().shape[0],
                               type(self._model.estimator).__name__,
                               *([(str(elt) if isinstance(elt, bool) else elt)
                                  if elt is not None else "nan"
                                  for elt in
                                  self._model.get_params_estimator().values()])
                               )
            print(report)
        else:

            report = (
                      "\033[94m================================================================================\033[0m\n"
                      "\033[1m\033[94m                         GRMlab Model Analysis                       \033[0m\n"
                      "\033[94m================================================================================\033[0m\n"
                      "\n"
                      " \033[1mGeneral information\033[0m\n"
                      "   model name                                  {:>30}\n"
                      "   number of records                           {:>30}\n"
                      "   number of variables                         {:>30}\n"
                      "   problem class                               {:>30}\n"
                      "   \033[94m--------------------------------------------------------------------------\033[0m\n"
                      "\n"
                      " \033[1mFeature Selection: Configuration Options\033[0m\n"
                      "   min features                                {:>30}\n"
                      "   max features                                {:>30}\n"
                      "   max correlation                             {:>30}\n"
                      "   abs correlation                             {:>30}\n"
                      "   excluded variables                          {:>30}\n"
                      "   forced variables                            {:>30}\n"
                      " \033[1mFeature Selection: Results\033[0m\n"
                      "   selected variables                          {:>30}\n"
                      "   infeasible variable pairs                   {:>30}\n"
                      "   group constraints                           {:>30}\n"
                      "   group constraints variables                 {:>30}\n"
                      "   \033[94m--------------------------------------------------------------------------\033[0m\n"
                      "\n"
                      " \033[1mEstimator: Configuration Options\033[0m\n"
                      "   estimator class                             {:>30}\n" +
                      "".join(["   " + elt.replace("_", " ") +
                              (44 - len(elt)) * " " + "{:>30}\n" for elt in
                              self._model.get_params_estimator().keys()]) +
                      "   \033[94m--------------------------------------------------------------------------\033[0m\n"
                      ).format(self._model.name,
                               self._model._n_samples,
                               self._model.feature_selection._solver._nvars,
                               self._model.feature_selection._solver.
                               _mipcl_problem.name,
                               self._model.feature_selection._solver.
                               n_min_features,
                               self._model.feature_selection._solver.
                               n_max_features,
                               self._model.feature_selection.max_correlation,
                               str(self._model.feature_selection.
                                   abs_correlation),
                               self._model.feature_selection._solver.
                               _n_excluded,
                               self._model.feature_selection._solver._n_fixed,
                               self._model.feature_selection._solver.
                               _n_selected_features,
                               self._model.feature_selection._solver.
                               _n_infeas_pairs,
                               self._model.feature_selection._solver._n_groups,
                               self._model.feature_selection._solver.
                               _n_groups_variables,
                               type(self._model.estimator).__name__,
                               *([(str(elt) if isinstance(elt, bool) else elt
                                   ) if elt is not None else "nan" for elt in
                                   self._model.get_params_estimator().values()]
                                 ))
            print(report)

    def stats(self, step="metrics"):
        """Model analysis results.

        Parameters
        ----------
        step : str (default="metrics")
            Options are "run", "metrics", and "features".
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        if step not in ("metrics", "features", "run"):
            raise ValueError("step not found.")

        if step == "run":
            if self.n_simulations is None:
                num_sim = np.nan
            else:
                num_sim = self.n_simulations

            report = (
                      "\033[94m================================================================================\033[0m\n"
                      "\033[1m\033[94m                         GRMlab Model Analysis: Run                          \033[0m\n"
                      "\033[94m================================================================================\033[0m\n"
                      "\n"
                      " \033[1mGeneral information                 Configuration options\033[0m\n"
                      "   number of samples   {:>8}        number of simulations            {:>5}\n"
                      "                                       simulation block size              {:>3.0%}\n"
                      "                                       feature analysis approximation   {:>5}\n"
                      "                                       temporal analysis                {:>5}\n"
                      "   \033[94m--------------------------------------------------------------------------\033[0m\n"
                      "\n"
                      " \033[1mTiming statistics\033[0m\n"
                      "   total                  {:>7.3f}\n"
                      "     metrics analysis     {:>7.3f} ({:>5.1%})\n"
                      "     feature analysis     {:>7.3f} ({:>5.1%})\n"
                      "   \033[94m--------------------------------------------------------------------------\033[0m\n"
                      ).format(self._n_samples, num_sim,
                               self.simulation_size,
                               str(self.feature_analysis_approximation),
                               str(self._is_dates_provided),
                               self._time_run, self._time_run_metrics,
                               self._time_run_metrics/self._time_run,
                               self._time_run_feature_analysis,
                               self._time_run_feature_analysis /
                               self._time_run)

            print(report)

        elif step == "metrics":
            self._metrics_stats()

        elif step == "features":
            report_data = []
            len_max = max([len(elt[0]) for elt in self._feature_importance] +
                          [5])
            val_max = max([elt[1] for elt in self._feature_importance])
            stars_num = int((74 - len_max - 6 - 8) / val_max)
            for name, val in self._feature_importance:
                report_data.append(name)
                report_data.append(self._vif[name])
                report_data.append(100 * val)
                report_data.append(int(round(val*stars_num, 0)) * "*")

            report = (
                      "\033[94m================================================================================\033[0m\n"
                      "\033[1m\033[94m                         GRMlab Model Analysis                       \033[0m\n"
                      "\033[94m================================================================================\033[0m\n"
                      "\n"
                      " \033[1mFeatures Analysis\033[0m\n"
                      "   Names" + " " * (len_max - 5) + "| VIF | Feature importance\n"
                      "   --------------------------------------------------------------------------\n" +
                      "".join(["   {:<" + str(len_max) + "}| {:3.2f}| {:4.1f}% {}\n"
                              for elt
                              in range(len(self._feature_importance))]) +
                      "   \033[94m--------------------------------------------------------------------------\033[0m\n"
                      ).format(*report_data)

            print(report)

            if hasattr(self._model.feature_selection, "max_correlation"):
                max_corr = self._model.feature_selection.max_correlation
            else:
                max_corr = 0.6

            if len(self._feature_names) > 1:
                # names for the feature selection
                plot_correlation_matrix(self._feature_names,
                                        max_corr,
                                        self._correlations)

    def _run(self, X, y, dates):
        time_init = time.perf_counter()

        self._metrics(X, y, dates)

        self._feature_analysis(X)

        self._time_run = time.perf_counter() - time_init

    def _feature_analysis(self, X):
        """Analysis of the features in the model.

        The analysis contains a feature importance analysis, a
        The feature analysis at this level is done using the test
        data. Some of these analysis may depend on other factors such as the
        estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
                The test input samples.
        """
        time_init = time.perf_counter()
        # variable names
        self._feature_names = self._model.get_support(output_names=True)
        if self._model.feature_selection is not None:
            X_trans = self._model.feature_selection.transform(X)
        else:
            X_trans = X
        # feature importance
        if self.verbose:
            print("   Feature Importance....")

        if hasattr(self._model.estimator, "coef_"):
            self._feature_coefficients = dict(zip(
                self._feature_names,
                self._model.estimator.coef_.ravel()
            ))
            if self.feature_analysis_approximation:
                self._linear_feature_importance(self._model.estimator.
                                                coef_.ravel(),
                                                X_trans)
            else:
                self._exact_feature_importance(X_trans)
        else:
            # default method
            self._exact_feature_importance(X_trans)

        if self.verbose:
            print("   Feature Importance Completed!\n   ---")

        # feature correlation
        if self.verbose:
            print("   Correlation Analysis....")

        self._correlations = np.abs(np.corrcoef(X_trans.T))
        if len(self._feature_names) > 2:
            corr = self._correlations - np.diag(np.diag(self._correlations))
            correlations_vector = corr[np.tril(corr) != 0.]
            self._correlation_metrics = basic_statistics(correlations_vector)
            gf = GaussianFilter(correlations_vector)
            self._correlation_metrics["histogram"] = {"y": list(gf._y_data),
                                                      "x": list(gf._x_data)}

        if self.verbose:
            print("   Correlation Analysis Completed!\n   ---")

        # Variance Inflation factor
        if self.verbose:
            print("   VIF calculation....")
        if np.shape(X_trans)[1] == 1:
            self._vif[self._feature_names[0]] = 0
            self._correlations = [[self._correlations]]
        else:
            self._variance_inflation_factor(X_trans)

        if self.verbose:
            print("   VIF calculation Completed!")

        self._time_run_feature_analysis = time.perf_counter() - time_init

    def _metrics(self, X, y, dates=None):
        """
        Analysis of model performance.

        It calculates the metrics related to the confusion matrix and other
        metrics as KS, Log-loss ....

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
                The test input samples.

        y : array-like, shape = [n_samples]
            The test target values.

        dates_test : array-like, shape = [n_samples]
                     or [n_dates] (default=None)
            Dates on the test data. This parameter is needed to implement
            a time analysis with multiple ModelAnalyzer results.
        """
        time_init = time.perf_counter()

        self._time_run_metrics = time.perf_counter() - time_init

    def _exact_feature_importance(self, X_trans):
        """
        Feature importance.

        Based on the standard deviation of the partial score of each feature.
        """
        n = X_trans.shape[1]
        X_mean = np.empty_like(X_trans)
        X_mean[:, :] = np.mean(X_trans, axis=0)

        stds = np.zeros(n)
        for j in range(n):
            col_j = X_mean[:, j].copy()
            X_mean[:, j] = X_trans[:, j]
            if hasattr(self._model.estimator, "predict_proba"):
                stds[j] = self._model.estimator.predict_proba(
                    X_mean)[:, 1].std()
            else:
                stds[j] = self._model.estimator.predict(X_mean).std()
            X_mean[:, j] = col_j

        if stds.sum() == 0:
            norm_weights = np.ones(n)/n
        else:
            norm_weights = stds / stds.sum()

        self._feature_importance = sorted(zip(self._feature_names,
                                              norm_weights),
                                          key=lambda x: x[1],
                                          reverse=True)

    def _linear_feature_importance(self, coef, X_trans):
        """
        Feature importance approximation.

        This approximation is exact for normal regression.
        """
        std_features = np.std(X_trans, axis=0)
        weights = np.abs(coef) * std_features
        norm_weights = weights / weights.sum()

        self._feature_importance = sorted(zip(self._feature_names,
                                              norm_weights),
                                          key=lambda x: x[1],
                                          reverse=True)

    def _variance_inflation_factor(self, X_trans):
        """
        Variance Inflation Factor of each independent variable.

        Independently of the used regression estimator, the VIF
        is done with linear regression.
        """
        for i in range(len(X_trans.T)):
            y_vif = X_trans.T[i]
            x_vif = np.delete(X_trans.T, i, 0).T
            reg = LinearRegression().fit(x_vif, y_vif)
            self._vif[self._feature_names[i]] = 1 / (1 - reg.score(x_vif,
                                                                   y_vif))

    def _init_dict(self):
        # Definition of the dictionary to store the results of the metrics.
        for k in _METRICS:
            self._dict_metrics[k] = DICT_STRUCT.copy()
            self._dict_metrics[k]["name"] = _METRICS[k]

        if self.n_simulations is not None:
            for k in _METRIC_RAND_COMPUTED:
                self._dict_metrics[k] = DICT_RANDOM.copy()
                self._dict_metrics[k]["name"] = _METRICS[k]
                self._dict_metrics[k]["values_rand"] = []

        if self._is_dates_provided:
            for k in _TEMPORAL_VARIABLES:
                self._dict_metrics[k]["temporal"] = []

    def _metrics_stats(self):
        pass

    def _check_rand_metric(self, name):
        if name not in _METRIC_RAND_COMPUTED:
            raise ValueError("metric {} is not supported.".format(name))

    def _check_temp_metric(self, name):
        if name not in _TEMPORAL_VARIABLES:
            raise ValueError("invalid metric for temporal analysis: {}."
                             .format(name))
