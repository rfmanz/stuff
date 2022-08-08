"""
Model Analyzer for continuous target
"""

# Authors: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2020.

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from scipy.stats import norm
from scipy.stats import pearsonr
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, normal_ad
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.weightstats import ztest

from ...core.exceptions import NotRunException
from ..metrics import regression_report
from .base import basic_statistics, DICT_STRUCT, DICT_RANDOM, ModelAnalyzerBase
from .plots import plot_multiclass_confusion_matrix, plot_sector_error
from .util import get_bins, year_intervals


_METRIC_RAND_COMPUTED = [
    'r2_score', 'mean_absolute_percentage_error', "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error", "pearson_corr", "spearson_rank_corr",
    "kendall_rank_corr"]

_TEMPORAL_VARIABLES = [
    'r2_score', 'mean_absolute_percentage_error', "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error", "pearson_corr", "spearson_rank_corr",
    "kendall_rank_corr"]

_METRICS = {
    'r2_score': "R2 score",
    'mean_absolute_percentage_error': 'MAPE',
    "mean_absolute_error": "Mean Absolute Error",
    "mean_squared_error": "Mean Squared Error",
    "root_mean_squared_error": "Root Mean Squared Error",
    "pearson_corr": "Pearson Correlation",
    "spearson_rank_corr": "Spearson Rank Correlation",
    "kendall_rank_corr": "Kendall Rank Correlation"
    }


def confusion_matrix(y_true, y_pred, bins=10, bins_pred=None):
    """
    Calculates the confusion matrix from a discretized continuous inference.

    Parameters
    ----------
    y_true : list or numpy array
        Ground truth (correct) target values.

    y_pred : list or numpy array
        Estimated target values.

    bins : int, list or numpy array (default=10)
        If bins is an int, it defines the number of equal populated bins to
        discretize the values. If bins is a list or numpy array, it has to be
        1-dimensional and monotonic; it will discretize the values with this
        cuts, allowing non-uniform bin population.

    bins_pred : None, int, list or numpy array (default=None)
        If None, it will get the value of parameter bins. If int, it defines
        the number of equal populated bins based on target true values to
        discretize the predicted values. If it is a list or numpy array, it has
        to be 1-dimensional and monotonic; it will discretize the values with
        this cuts, allowing non-uniform bin population.

    Returns
    -------
    names : list
        The elements of the list names contains the description of each bin, in
        the form: "[initial_value, final_value)". The order of names coincides
        with count_hist.

    count_hist : np.array
        Two dimensional numpy array with the confusion matrix. The order of the
        columns and rows of count_hist coincide with names.
    """
    n_bins, bins = get_bins(y_true, bins)
    if bins_pred is None:
        n_bins_pred = n_bins
        bins_pred = bins
    else:
        n_bins_pred, bins_pred = get_bins(y_true, bins_pred)

    # discretization of y_true and y_pred
    inds_true = np.digitize(y_true, bins)
    inds_pred = np.digitize(y_pred, bins_pred)

    # generation of confusion matrix
    count_hist = np.histogram2d(
        x=inds_true, y=inds_pred, bins=[np.arange(0, n_bins+1, 1),
                                        np.arange(0, n_bins_pred+1, 1)])[0]

    # description of the limits of each bin
    if n_bins == 1:
        names_t = ["All"]
    else:
        names_t = []
        names_t += ["[..., " + str(bins[0]) + ")"]
        names_t += ["[" + str(elt) + ", " + str(bins[i+1]) + ")"
                    for i, elt in enumerate(bins[:-1])]
        names_t += ["[" + str(bins[-1]) + ", ...)"]

    if n_bins_pred == 1:
        names_p = ["All"]
    else:
        names_p = []
        names_p += ["[..., " + str(bins_pred[0]) + ")"]
        names_p += ["[" + str(elt) + ", " + str(bins_pred[i+1]) + ")"
                    for i, elt in enumerate(bins_pred[:-1])]
        names_p += ["[" + str(bins_pred[-1]) + ", ...)"]

    return (names_p, names_t), count_hist


def sector_error(y_true, y_pred, bins=10, error_delta=0.3):
    """
    Calculates the relative error in relation with the target value.

    Parameters
    ----------
    y_true : list or numpy array
        Ground truth (correct) target values.

    y_pred : list or numpy array
        Estimated target values.

    bins : int, list or numpy array (default=10)
        If bins is an int, it defines the number of equal populated bins to
        discretize the values. If bins is a list or numpy array, it has to be
        1-dimensional and monotonic; it will discretize the values with this
        cuts, allowing non-uniform bin population.

    error_delta : float (default=0.3)
        Discretization of the relative error values. The user can define the
        width of the discret bins.

    Returns
    -------
    (names_x, names_y) : (list, list)
        The elements of the tuple contains the description of each
        2-dimensional bin. In names_x: "[target_initial_value,
        target_final_value)". In names_y: "[relative_error_initial_value,
        relative_error_final_value)". The order coincides with count_hist.

    count_hist : np.array
        Two dimensional numpy array with the 2-dim histogram. The order of the
        columns and rows of count_hist coincide with names.
    """

    # last relative error cut for discretization of errors
    last_point = (np.floor((1-error_delta)/error_delta)*error_delta +
                  error_delta/2)

    # relative errors
    error = (y_pred-y_true)/y_true
    bins_error = np.round(np.arange(
        -last_point, last_point+error_delta, error_delta), 4)

    # discretization of relative errors
    inds_error = np.digitize(error, bins_error)

    n_bins, bins = get_bins(y_true, bins)

    # discretization of y_true
    inds_true = np.digitize(y_true, bins)

    # histogram
    count_hist = np.histogram2d(
        x=inds_true, y=inds_error,
        bins=[np.arange(0, n_bins+1, 1),
              np.arange(0, len(bins_error)+2, 1)])[0]

    # description of the limits of each bin for y_true
    if n_bins == 1:
        names_x = ["All"]
    else:
        names_x = []
        names_x += ["[..., " + str(bins[0]) + ")"]
        names_x += ["[" + str(elt) + ", " + str(bins[i+1]) + ")"
                    for i, elt in enumerate(bins[:-1])]
        names_x += ["[" + str(bins[-1]) + ", ...)"]

    # description of the limits of each bin for the relative errors
    names_y = []
    names_y += ["[..., " + str(bins_error[0]) + ")"]
    names_y += ["[" + str(elt) + ", " + str(bins_error[i+1]) + ")"
                for i, elt in enumerate(bins_error[:-1])]
    names_y += ["[" + str(bins_error[-1]) + ", ...)"]

    return (names_x, names_y), count_hist


def sector_metrics(y_true, y_pred, bins):
    """
    Calculates metrics for each bin.

    Parameters
    ----------
    y_true : list or numpy array
        Ground truth (correct) target values.

    y_pred : list or numpy array
        Estimated target values.

    bins : int, list or numpy array (default=10)
        If bins is an int, it defines the number of equal populated bins to
        discretize the values. If bins is a list or numpy array, it has to be
        1-dimensional and monotonic; it will discretize the values with this
        cuts, allowing non-uniform bin population.

    Returns
    -------
    df_errors : pandas dataframe
        Error metric for each bin. The key of the bin is the index of the
        dataframe.

    df_basic_stats : pandas dataframe
        Basics metric for each bin. The key of the bin is the index of the
        dataframe.

    df_corrs : pandas dataframe
        Correlations within each bin. The key of the bin is the index of the
        dataframe.
    """
    n_bins, bins = get_bins(y_true, bins)

    inds_true = np.digitize(y_true, bins)

    metrics = [
        "mean_error",
        "median_error", "std_error",
        "mean_absolute_error",
        "median_absolute_error",
        "mean_absolute_percentage_error", "mean_percentage_error",
        "median_percentage_error", "std_percentage_error"]

    corrs = ["spearson_rank_corr", "kendall_rank_corr", "pearson_corr"]

    # results of error metrics
    results = {}

    # basic statistics
    mean_vec = []
    median_vec = []
    std_vec = []
    mean_true_vec = []
    median_true_vec = []
    std_true_vec = []

    # metric calculation in a loop for each bin
    for cat in range(n_bins):
        y_true_set = y_true[inds_true == cat]
        y_pred_set = y_pred[inds_true == cat]
        res_metrics = regression_report(y_true_set, y_pred_set,
                                        metrics=metrics+corrs)
        results[cat] = res_metrics

        mean_vec.append(np.mean(y_pred_set))
        median_vec.append(np.median(y_pred_set))
        std_vec.append(np.std(y_pred_set))

        mean_true_vec.append(np.mean(y_true_set))
        median_true_vec.append(np.median(y_true_set))
        std_true_vec.append(np.std(y_true_set))

    # description of the limits of each bin
    if n_bins == 1:
        names = ["All"]
    else:
        names = []
        names += ["[..., " + str(bins[0]) + ")"]
        names += ["[" + str(elt) + ", " + str(bins[i+1]) + ")"
                  for i, elt in enumerate(bins[:-1])]
        names += ["[" + str(bins[-1]) + ", ...)"]

    # error dataframe
    df_errors = pd.DataFrame.from_dict(results).T[metrics]
    df_errors.index = names

    # corr dataframe
    df_corrs = pd.DataFrame.from_dict(results).T[corrs]
    df_corrs.index = names

    # basic stats dataframe
    df_basic_stats = pd.DataFrame(index=names)
    df_basic_stats["count"] = np.unique(inds_true, return_counts=True)[1]
    df_basic_stats["mean_pred"] = mean_vec
    df_basic_stats["median_pred"] = median_vec
    df_basic_stats["std_pred"] = std_vec
    df_basic_stats["mean_true"] = mean_true_vec
    df_basic_stats["median_true"] = median_true_vec
    df_basic_stats["std_true"] = std_true_vec

    return df_errors, df_basic_stats, df_corrs


class LinearAssumptionsTest():
    """Analysis of the assumptions of Gauss–Markov theorem for OLS regression.

    Parameters
    ----------
    model : :class:`GRMlabModel`
        The fitted model to be analyzed.

    alpha : float (default=0.05)
        Max p_value to reject the null hypothesis.
    """

    def __init__(self, model, alpha=0.05):
        self.model = model
        self.alpha = alpha

        self.dict_tests = {}
        self.y = None
        self.y_pred = None
        self.residuals = None
        self.stats_residuals = None

        self._is_run = False

    def run(self, X, y):
        """
        Runs all the tests.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
                The test input samples.

        y : array-like, shape = [n_samples]
            The test target values.
        """
        self._run(X, y)

        self._is_run = True

    def stats(self):
        """Print the result of the analysis.
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        report = (
                  "\033[94m================================================================================\033[0m\n"
                  "\033[1m\033[94m                 GRMlab Gauss–Markov theorem Assumptions                       \033[0m\n"
                  "\033[94m================================================================================\033[0m\n"
                  "\n"
                  " \033[1m                    Residuals Stats\033[0m\n"
                  "             Mean                   {:>8.7f}\n"
                  "             Standard Deviation     {:>8.7f}\n"
                  "             Skewness               {:>8.7f}\n"
                  "             Excess Kurtosis        {:>8.7f}\n"
                  "             ---\n"
                  "             P-99                   {:>8.7f}\n"
                  "             P-95                   {:>8.7f}\n"
                  "             P-75                   {:>8.7f}\n"
                  "             Median                 {:>8.7f}\n"
                  "             P-25                   {:>8.7f}\n"
                  "             P-5                    {:>8.7f}\n"
                  "             P-1                    {:>8.7f}\n"
                  "\n"
                  " \033[1mLinearity\033[0m\n"
                  "     Check on the True vs. predicted scattered plot. Predictions should\n"
                  "     follow the diagonal line\n"
                  "\n"
                  " \033[1mResiduals Zero Mean\033[0m\n"
                  "     Z-Test:                {:>1.5f}     p_value: {:>1.7f}\n"
                  "        Result: {}\n"
                  "\n"
                  " \033[1mExogenity\033[0m\n"
                  "     Pearson-Test:          {:>1.5f}     p_value: {:>1.7f}\n"
                  "        Result: {}\n"
                  "\n"
                  " \033[1mResiduals Normality (not required but recommended)\033[0m\n"
                  "     Anderson Darling Test: {:>1.5f}     p_value: {:>1.7f}\n"
                  "        Result: {}\n"
                  "\n"
                  " \033[1mMulticollinearity\033[0m\n"
                  "     Check VIF on method .stats('features') of ModelAnalyzerContinuous\n"
                  "        OK range: [1.0, 2.6], best: 1.0\n"
                  "\n"
                  " \033[1mResiduals Autocorrelation (important for time series models)\033[0m\n"
                  "     Durbin Watson Test:    {:>1.5f}\n"
                  "        OK range: [1.5, 2.5], best: 2.0\n"
                  "\n"
                  " \033[1mHeteroskedasticity Tests\033[0m\n"
                  "     White Test:            {:>1.5f}     p_value: {:>1.7f}\n"
                  "        Result: {}\n"
                  "     Breusch–Pagan Test:    {:>1.5f}     p_value: {:>1.7f}\n"
                  "        Result: {}\n"
                  "\033[94m--------------------------------------------------------------------------\033[0m\n"
                  ).format(self.stats_residuals['mean_rand'],
                           self.stats_residuals['std_rand'],
                           self.stats_residuals['skew_rand'],
                           self.stats_residuals['kurtosis_rand'],
                           self.stats_residuals['p99_rand'],
                           self.stats_residuals['p95_rand'],
                           self.stats_residuals['p75_rand'],
                           self.stats_residuals['median_rand'],
                           self.stats_residuals['p25_rand'],
                           self.stats_residuals['p5_rand'],
                           self.stats_residuals['p1_rand'],
                           self.dict_tests['z_test']['value'],
                           self.dict_tests['z_test']['p_value'],
                           self._test_result(self.dict_tests[
                            'z_test']['p_value']),
                           self.dict_tests['corr_pearson_test']['value'],
                           self.dict_tests['corr_pearson_test']['p_value'],
                           self._test_result(self.dict_tests[
                            'corr_pearson_test']['p_value']),
                           self.dict_tests['anderson_darling_test']['value'],
                           self.dict_tests['anderson_darling_test']['p_value'],
                           self._test_result(self.dict_tests[
                            'anderson_darling_test']['p_value']),
                           self.dict_tests['durbin_watson_test']['value'],
                           self.dict_tests['white_test']['value'],
                           self.dict_tests['white_test']['p_value'],
                           self._test_result(self.dict_tests[
                            'white_test']['p_value']),
                           self.dict_tests['breusch_pagan_test']['value'],
                           self.dict_tests['breusch_pagan_test']['p_value'],
                           self._test_result(self.dict_tests[
                            'breusch_pagan_test']['p_value']))
        print(report)

        self._plot_linear_relation()

        self._plot_Q_Q()

        self._plot_ordered_residuals()

        self._plot_residuals_predicted()

    def _exogenity_assumption(self):
        """
        Check the exogenity assumption. When it fails, it is called
        endogenity.

        Consequences of failure:
            * bias on the coefficient estimate.
            * OLS incorrectly attributes some of the variance explained by the
              error term to the independent variable.

        Causes of error:
            * Omitted variable bias.
            * Simultaneity bias between the independent and dependent
              variables.
            * Measurement error in the independent variables.

        How to solve if it is not satisfied:
            * Add new variables.
            * Instrumental variables.
            * Interaction terms.
        """
        try:
            corr_test = pearsonr(self.y_pred, self.residuals)
        except:
            corr_test = [np.nan, np.nan]
        self.dict_tests["corr_pearson_test"] = {
            "value": corr_test[0], "p_value": corr_test[1]}

    def _homoskedasticity(self, X):
        """
        Check the homoskedasticity assumption of residuals variance.

        Consequences of failure:
            * Reduces the precision of the estimates in OLS linear regression.
            * Model gives too much weight to the subset of the data with higher
              residuals variance.

        How to solve if it is not satisfied:
            * Weighting the records using weighted OLS.
            * Transform highly skewed variables dependent or independent
              (ex: log transformation).

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The test input samples.
        """
        X_exog = np.append(np.full((X.shape[0], 1), 1), X, axis=1)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                white_test = het_white(self.residuals, X_exog)
        except:
            white_test = [np.nan, np.nan]
        self.dict_tests["white_test"] = {
            "value": white_test[0], "p_value": white_test[1]}

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                BP_test = het_breuschpagan(self.residuals, X_exog)
        except:
            BP_test = [np.nan, np.nan]
        self.dict_tests["breusch_pagan_test"] = {
            "value": BP_test[0], "p_value": BP_test[1]}

    def _linear_assumption(self):
        """
        Check the linear assumption of predictors vs. target.

        Consequences of failure:
            * Inaccurate predictions: Under fitting model

        How to solve if it is not satisfied:
            * Add polynomial terms.
            * Perform non-linear transformations on predictors.
            * Add new variables.
        """
        raise NotImplementedError

    def _multicolineraity_predictors(self):
        """
        Check the no-multicoliniearity assumption among predictors.

        Consequences of failure:
            * Increases the standard error of the coefficients.
            * Issues with the interpretation of the coefficients.

        How to solve if it is not satisfied:
            * Remove predictors with high VIF.
            * Perform dimensionality reduction.
        """
        raise NotImplementedError

    def _plot_linear_relation(self):
        plt.figure(figsize=(15/2, 10/2))
        plt.scatter(self.y, self.y_pred, color="#2dcccd", label="prediction")
        plt.plot([min(self.y), max(self.y)], [min(self.y), max(self.y)],
                 "-.", color="#1464A5", label="perfect prediction",
                 linewidth=1)
        plt.legend()
        plt.title("True vs. predicted values")
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.show()
        plt.close()

    def _plot_ordered_residuals(self):
        plt.figure(figsize=(15/2, 10/2))
        plt.plot(self.residuals, color="#2dcccd", label="residuals evolution")
        plt.plot([0, len(self.residuals)], [0, 0], "-.", color="#1464A5",
                 linewidth=1, label="zero residuals")
        plt.xlim(0, len(self.residuals))
        plt.legend()
        plt.title("Residuals in order")
        plt.xlabel("Order as collected by model")
        plt.ylabel("Residuals")
        plt.show()
        plt.close()

    def _plot_Q_Q(self, bin_width=0.5):
        plt.figure(figsize=(15/2, 10/2))
        std_errors = np.std(self.residuals)
        y_Q = np.percentile((self.residuals)/std_errors,
                            np.arange(bin_width, 100, bin_width))
        x_Q = norm.ppf(np.arange(bin_width/100, 1, bin_width/100),
                       loc=0, scale=1)
        plt.scatter(x_Q, y_Q, color="#2dcccd", label="actual dist.")
        plt.plot([-3, 3], [-3, 3], "-.", color="#1464A5", linewidth=1,
                 label="perfect normality")
        plt.legend()
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
        plt.title("Residuals Q-Q plot")
        plt.show()
        plt.close()

    def _plot_residuals_predicted(self):
        plt.figure(figsize=(15/2, 10/2))
        plt.scatter(self.y_pred, self.residuals, color="#2dcccd",
                    label="residuals")
        plt.plot([min(self.y_pred), max(self.y_pred)], [0, 0],
                 "-.", color="#1464A5", label="zero residuals",
                 linewidth=1)
        plt.legend()
        plt.title("Predicted values vs. residuals")
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals")
        plt.show()
        plt.close()

    def _residuals_normality(self):
        """
        Check the residuals normality assumption. (Not required but
        recommended)

        Consequences of failure:
            * Non-reliable confidence intervals and prediction intervals.

        How to solve if it is not satisfied:
            * Remove outliers.
            * Perform non-linear transformations on predictors.
            * Exclude some specific variables (ex: long-tailed ones).
        """
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                ad_test = normal_ad(self.residuals)
        except:
            ad_test = [np.nan, np.nan]
        self.dict_tests["anderson_darling_test"] = {
            "value": ad_test[0], "p_value": ad_test[1]}

    def _residuals_zero_mean(self):
        """
        Check the zero mean residuals assumption.

        Consequences of failure:
            * Systematic error.
            * Part of the error is predictable, i.e., not random.

        How to solve if it is not satisfied:
            * Add a constant term to the regression.
        """
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                z_test = ztest(self.residuals, value=0.0,
                               alternative='two-sided')
        except:
            z_test = [np.nan, np.nan]
        self.dict_tests["z_test"] = {
            "value": z_test[0], "p_value": z_test[1]}

    def _residuals_autocorrelations(self):
        """
        Check the no correlation assumption of residuals.

        Consequences of failure:
            * Under/over predicting certain conditions.
            * Past information not being captured.

        How to solve if it is not satisfied:
            * Add lag variables.
            * Add an independent variable.
            * Interaction terms and/or other variable transformations.
        """
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                dw_test = durbin_watson(self.residuals)
        except:
            dw_test = np.nan
        self.dict_tests["durbin_watson_test"] = {
            "value": dw_test, "p_value": None}

    def _run(self, X, y):

        self.y = y
        self.y_pred = self.model.predict(X)
        self.residuals = (self.y-self.y_pred)

        # basic statistics for the residual distribution
        self.stats_residuals = basic_statistics(self.residuals)

        # Test for all linear regression assumptions
        self._residuals_zero_mean()
        self._exogenity_assumption()
        self._residuals_normality()
        self._residuals_autocorrelations()
        self._homoskedasticity(X)

    def _test_result(self, p_value):
        if np.isnan(p_value):
            return "Undefined"
        elif p_value > self.alpha:
            return 'Assumption satisfied'
        else:
            return 'Assumption not satisfied'


class ModelAnalyzerContinuous(ModelAnalyzerBase):
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

    feature_analysis_approximation : Boolean (default=False)
        If True, it calculates only first order approximations for
        non-analytical calculations.

    bins : int, list or numpy array (default=10)
        If bins is an int, it defines the number of equal populated bins to
        discretize the values based on target true values. If bins is a list or
        numpy array, it has to be 1-dimensional and monotonic; it will
        discretize the values with this cuts, allowing non-uniform bin
        population.

    bins_pred : None, int, list or numpy array (default=None)
        If None, it will get the value of parameter bins. If int, it defines
        the number of equal populated bins based on target true values to
        discretize the predicted values. If it is a list or numpy array, it has
        to be 1-dimensional and monotonic; it will discretize the values with
        this cuts, allowing non-uniform bin population.

    error_delta : float (default=0.3)
        Discretization of the relative error values. The user can define the
        width of the discrete bins.

    verbose : Boolean (default=True)
        Controls verbosity of output.
    """
    def __init__(self, name, model, n_simulations=100, simulation_size=1.,
                 seed=None, feature_analysis_approximation=False, bins=10,
                 bins_pred=None, error_delta=0.3, verbose=True):

        # model variables
        self.name = name
        self.model = model
        self.n_simulations = n_simulations
        self.simulation_size = simulation_size
        self.seed = seed
        self.feature_analysis_approximation = feature_analysis_approximation
        self.bins = bins
        self.bins_pred = bins_pred
        self.error_delta = error_delta
        self.verbose = verbose

        if self.bins_pred is None:
            self.bins_pred = self.bins

        # data
        self._n_samples = None
        self._n_columns = None

        # model variables
        self._model = None
        self._optimizer = None
        self._model_instance = None

        # metrics
        self.n_bins = None
        self._dict_metrics = {}
        self.df_bin_errors = None
        self.df_bin_basics = None
        self._linear_assumptions = None

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

        # timing
        self._time_run = None
        self._time_run_metrics = None
        self._time_run_feature_analysis = None

        # flags
        self._is_dates_provided = False
        self._is_run = False

    def sector_metrics(self):
        """
        Display the analysis with discretized target.

        Plot metrics related to the multicategorical confusion matrix,
        error metrics, basic statistics and correlations for each discretized
        category.

        Returns
        -------
        df_bin_errors : pandas dataframe
            Error metrics for category. The key of the bin is the index of the
            dataframe.

        df_bin_basics : pandas dataframe
            Basics statistics for category. The key of the bin is the index of
            the dataframe.

        df_corrs : pandas dataframe
            Correlations within each category. The key of the bin is the index
            of the dataframe.
        """

        # Confusion Matrix
        plot_multiclass_confusion_matrix(
            self.conf_mat_names[0], self.conf_mat_names[1], self.conf_mat)

        # Sector error
        plot_sector_error(self.error_mat_names[0], self.error_mat_names[1],
                          self.error_mat)

        self._plot_basic_bin_metrics()

        self._plot_error_bin_metrics()

        return self.df_bin_errors, self.df_bin_basics, self.df_corrs

    def _check_rand_metric(self, name):
        if name not in _METRIC_RAND_COMPUTED:
            raise ValueError("metric {} is not supported.".format(name))

    def _check_temp_metric(self, name):
        if name not in _TEMPORAL_VARIABLES:
            raise ValueError("invalid metric for temporal analysis: {}."
                             .format(name))

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
        y_pred = self._model.predict(X)

        self.conf_mat_names, self.conf_mat = confusion_matrix(
            y_true=y, y_pred=y_pred, bins=self.bins, bins_pred=self.bins_pred)

        self.error_mat_names, self.error_mat = sector_error(
            y_true=y, y_pred=y_pred, bins=self.bins,
            error_delta=self.error_delta)

        self.df_bin_errors, self.df_bin_basics, self.df_corrs = sector_metrics(
            y_true=y, y_pred=y_pred, bins=self.bins)

        self.n_bins = self.df_bin_errors.shape[0]

        # Linear assumptions
        self._linear_assumptions = LinearAssumptionsTest(model=self._model)
        self._linear_assumptions.run(X, y)

        # Metrics derivation
        calc_metrics = regression_report(y_true=y, y_pred=y_pred,
                                         metrics=list(_METRICS.keys()),
                                         output_dict=True)

        for name, value in calc_metrics.items():
            self._dict_metrics[name]['global_value'] = value

        if self.verbose:
            print("   Main metrics and curves obtained!\n   ---")

        # simulations
        if self.n_simulations is not None:
            np.random.seed(self.seed)
            if self.verbose:
                print("   Starting simulations....")

            for i in range(self.n_simulations):
                idx = np.random.choice(np.arange(self._n_samples),
                                       int(self.simulation_size *
                                           self._n_samples),
                                       replace=True)
                y_pred_rand = y_pred[idx]
                y_test_rand = y[idx]

                sim_result = regression_report(
                    y_true=y_test_rand,
                    y_pred=y_pred_rand,
                    metrics=_METRIC_RAND_COMPUTED
                )

                for mtr in _METRIC_RAND_COMPUTED:
                    self._dict_metrics[mtr]['values_rand'].append(
                                                            sim_result[mtr])
                if self.verbose and (i+1) % 100 == 0:
                    print("      {} simulations executed".format(i+1))

            for mtr in _METRIC_RAND_COMPUTED:
                metrics_aux = basic_statistics(
                    self._dict_metrics[mtr]['values_rand'])
                for k_aux in metrics_aux.keys():
                    self._dict_metrics[mtr][k_aux] = metrics_aux[k_aux]
            if self.verbose:
                print("   Simulations concluded!\n   ---")

        # Time analysis
        if self._is_dates_provided:
            if self.verbose:
                print("   Starting time analysis....")
            self._temporal_analysis(y, y_pred, dates)

            if self.verbose:
                print("   Time analysis completed!")

        self._time_run_metrics = time.perf_counter() - time_init

    def _metrics_stats(self):
        report = (
                  "\033[94m================================================================================\033[0m\n"
                  "\033[1m\033[94m                         GRMlab Model Analysis                       \033[0m\n"
                  "\033[94m================================================================================\033[0m\n"
                  "\n"
                  " \033[1mMetrics\033[0m\n"
                  "     R2                   {:>1.5f} +/- {:>1.5f} (95%); std: {:>1.7f}\n"
                  "     MAPE                 {:>1.5f} +/- {:>1.5f} (95%); std: {:>1.7f}\n"
                  "     RMSE                 {:>1.5f} +/- {:>1.5f} (95%); std: {:>1.7f}\n"
                  "     MSE                  {:>1.5f} +/- {:>1.5f} (95%); std: {:>1.7f}\n"
                  "     MAE                  {:>1.5f} +/- {:>1.5f} (95%); std: {:>1.7f}\n"
                  "\n"
                  " \033[1mRank Correlations\033[0m\n"
                  "     Kendall rank C.      {:>1.5f} +/- {:>1.5f} (95%); std: {:>1.7f}\n"
                  "     Spearman's rank C.   {:>1.5f} +/- {:>1.5f} (95%); std: {:>1.7f}\n"
                  " \033[1mCorrelations\033[0m\n"
                  "     Pearson Correlation  {:>1.5f} +/- {:>1.5f} (95%); std: {:>1.7f}\n"
                  "\033[94m--------------------------------------------------------------------------\033[0m\n"
                  ).format(self._dict_metrics['r2_score']['global_value'],
                           self._dict_metrics['r2_score']['std_rand'] * 1.96,
                           self._dict_metrics['r2_score']['std_rand'],
                           self._dict_metrics['mean_absolute_percentage_error']['global_value'],
                           self._dict_metrics['mean_absolute_percentage_error']['std_rand'] * 1.96,
                           self._dict_metrics['mean_absolute_percentage_error']['std_rand'],
                           self._dict_metrics['root_mean_squared_error'][
                            'global_value'],
                           self._dict_metrics['root_mean_squared_error'][
                            'std_rand'] * 1.96,
                           self._dict_metrics['root_mean_squared_error'][
                            'std_rand'],
                           self._dict_metrics['mean_squared_error'][
                            'global_value'],
                           self._dict_metrics['mean_squared_error'][
                            'std_rand'] * 1.96,
                           self._dict_metrics['mean_squared_error'][
                            'std_rand'],
                           self._dict_metrics['mean_absolute_error'][
                            'global_value'],
                           self._dict_metrics['mean_absolute_error'][
                            'std_rand'] * 1.96,
                           self._dict_metrics['mean_absolute_error'][
                            'std_rand'],
                           self._dict_metrics['kendall_rank_corr'][
                            'global_value'],
                           self._dict_metrics['kendall_rank_corr'][
                            'std_rand'] * 1.96,
                           self._dict_metrics['kendall_rank_corr']['std_rand'],
                           self._dict_metrics['spearson_rank_corr'][
                            'global_value'],
                           self._dict_metrics['spearson_rank_corr'][
                            'std_rand'] * 1.96,
                           self._dict_metrics['spearson_rank_corr'][
                            'std_rand'],
                           self._dict_metrics['pearson_corr']['global_value'],
                           self._dict_metrics['pearson_corr'][
                            'std_rand'] * 1.96,
                           self._dict_metrics['pearson_corr']['std_rand'])
        print(report)

        self._linear_assumptions.stats()

    def _plot_basic_bin_metrics(self):
        mean_values = self.df_bin_basics["mean_pred"].values
        mean_true = self.df_bin_basics["mean_true"].values
        error_sup = mean_values + 1.95*(self.df_bin_basics["std_pred"].values)
        error_low = mean_values - 1.95*(self.df_bin_basics["std_pred"].values)
        error_true_sup = mean_true + 1.95*(
            self.df_bin_basics["std_true"].values)
        error_true_low = mean_true - 1.95*(
            self.df_bin_basics["std_true"].values)

        plt.figure(figsize=(15/2, 10/2))

        plt.plot(self.df_bin_basics.index.values, mean_values, marker="o",
                 c="#1464A5", label="mean predicted")
        plt.plot(self.df_bin_basics.index.values, error_sup, "--", marker=".",
                 c="#1464A5", linewidth=1, label="$\pm 1.95\sigma$ predicted")
        plt.plot(self.df_bin_basics.index.values, error_low, "--", marker=".",
                 c="#1464A5", linewidth=1)
        plt.fill_between(self.df_bin_basics.index.values, error_sup, error_low,
                         facecolor='#1464A5', alpha=0.05)

        plt.plot(self.df_bin_basics.index.values, mean_true, marker="o",
                 c="#02A5A5", label="mean true")
        plt.plot(self.df_bin_basics.index.values, error_true_sup, "--",
                 marker=".", c="#02A5A5", linewidth=1,
                 label="$\pm 1.95\sigma$ true")
        plt.plot(self.df_bin_basics.index.values, error_true_low, "--",
                 marker=".", c="#02A5A5", linewidth=1)
        plt.fill_between(self.df_bin_basics.index.values, error_true_sup,
                         error_true_low, facecolor='#02A5A5', alpha=0.05)

        plt.title("True and Predicted value by sector")
        plt.legend()
        plt.xticks(rotation=90)
        plt.show()
        plt.close()

    def _plot_error_bin_metrics(self):

        mean_error_values = self.df_bin_errors["mean_error"].values
        error_sup = mean_error_values + 1.95*(
            self.df_bin_errors["std_error"].values)
        error_low = mean_error_values - 1.95*(
            self.df_bin_errors["std_error"].values)

        plt.figure(figsize=(15/2, 10/2))

        plt.plot(self.df_bin_errors.index.values, mean_error_values,
                 marker="o", c="#F35E61", label="mean error")
        plt.plot(self.df_bin_errors.index.values, error_sup, "--",
                 marker=".", c="#F35E61", linewidth=1,
                 label="$\pm 1.95\sigma$")
        plt.plot(self.df_bin_errors.index.values, error_low, "--",
                 marker=".", c="#F35E61", linewidth=1)
        plt.fill_between(self.df_bin_errors.index.values, error_sup,
                         error_low, facecolor='#F35E61', alpha=0.05)
        plt.plot([0, self.n_bins-1], [0, 0], "r-.",
                 c="#1464A5", linewidth=1, alpha=0.6)

        plt.title("Errors by sector")
        plt.legend()
        plt.xticks(rotation=90)
        plt.show()
        plt.close()

    def _temporal_analysis(self, y, y_pred, dates):
        """Temporal analysis of model performance.

        It calculates the performance metrics of the model for 1-year windows.

        Parameters
        ----------
        y : array-like, shape = [n_samples]
            The target values.

        y_pred : array-like, shape = [n_samples]
            The target values predicted by the model.

        dates: array-like, shape = [n_samples] or [n_dates] (default=None)
            Dates of the observations
        """
        intervals = year_intervals(min(dates), max(dates))
        flag_first = True

        for interval in intervals:
            ind = np.logical_and(np.greater_equal(dates, interval[0]),
                                 np.greater(interval[1], dates))

            y_ind = y[ind]
            y_pred_ind = y_pred[ind]

            if len(y_ind) != 0 and (sum(y_ind) != 0) and \
               (sum(y_pred_ind) != 0) and (len(y_ind)-sum(y_ind) != 0):
                aux_metrics = regression_report(
                    y_true=y_ind, y_pred=y_pred_ind,
                    metrics=_TEMPORAL_VARIABLES
                )
                # In the first execution, inicializate to empty lists
                if flag_first:
                    self._count_temporal = []
                    for k in aux_metrics.keys():
                        if k in self._dict_metrics.keys():
                            self._dict_metrics[k]['temporal'] = []
                        else:
                            self._dict_metrics[k] = {}
                            self._dict_metrics[k]['temporal'] = []
                    flag_first = False
                for k in aux_metrics.keys():
                    self._dict_metrics[k]['temporal'].append(aux_metrics[k])
                self._count_temporal.append(len(y_ind))
            elif not flag_first:
                for k in aux_metrics.keys():
                    self._dict_metrics[k]['temporal'].append(None)
                self._count_temporal.append(None)

        self._intervals_temporal = intervals
