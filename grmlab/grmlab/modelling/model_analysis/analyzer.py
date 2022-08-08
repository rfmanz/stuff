"""
Model Analyzer
"""

# Authors: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
#          Carlos González Berrendero <c.gonzalez.berrender@bbva.com>
# BBVA - Copyright 2019.

import time
import numpy as np

from sklearn.metrics import roc_curve

from ...core.exceptions import NotRunException
from ..metrics import binary_classification_report
from .base import basic_statistics, DICT_STRUCT, DICT_RANDOM, ModelAnalyzerBase
from .plots import plot_roc
from .util import year_intervals


_METRIC_RAND_COMPUTED = ['gini', 'kappa', 'mcc']

_TEMPORAL_VARIABLES = [
    'gini', 'default_rate', 'balanced_accuracy', 'balanced_error_rate',
    'diagnostic_odds_ratio', 'discriminant_power', 'fnr', 'fpr',
    'geometric_mean', 'positive_likelihood', 'tpr', 'tnr', 'youden']

_METRICS = {
    'tp': "True positives", 'tn': "True negatives",
    'fp': "False positives", 'fn': "False negatives",
    'tpr': "True positive rates", 'tnr': "True negative rates",
    'fpr': "False positive rates", 'fnr': "False negative rates",
    'youden': "Youden index", 'accuracy': "Accuracy",
    'ppv': "Positive predictive value (PPV)",
    'npv': "Negative predictive value (PPV)",
    'lift_p': "Lift+", 'lift_n': "Lift-", 'gini': "Gini",
    'kappa': "Cohen-Kappa score", 'mcc': "Matthews correlation coefficient",
    'log_loss': "Log-loss",
    'balanced_accuracy': "Balanced accuracy",
    'balanced_error_rate': "Balanced error rate",
    'diagnostic_odds_ratio': "Diagnostic odds ratio",
    'discriminant_power': "Discriminant power",
    'geometric_mean': "Geometric mean",
    'negative_likelihood': "Negative likelihood",
    'positive_likelihood': "Positive likelihood",
    'default_rate': "Default rate"
    }


class ModelAnalyzer(ModelAnalyzerBase):
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

    def plot_roc(self):
        """Plots the ROC curve.
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        plot_roc(*self._roc)

    def _metrics_stats(self):
        report = (
                  "\033[94m================================================================================\033[0m\n"
                  "\033[1m\033[94m                         GRMlab Model Analysis                       \033[0m\n"
                  "\033[94m================================================================================\033[0m\n"
                  "\n"
                  " \033[1mMetrics\033[0m\n"
                  "              Actual values\n"
                  " P           1      |       0                Precision        Lift\n"
                  " r    -----------------------------\n"
                  " e 1 | TP: {:>8} | FP: {:>8}           {:>1.5f}          {:>7.5f}\n"
                  " d   | TPR: {:>1.5f} | FPR: {:>1.5f}\n"
                  " i    -----------------------------\n"
                  " c 0 | FN: {:>8} | TN: {:>8}           {:>1.5f}          {:>7.5f}\n"
                  " t   | FNR: {:>1.5f} | TNR: {:>1.5f}\n"
                  " e   ------------------------------\n"
                  " d\n"
                  "     Accuracy       {:>1.5f}\n"
                  "     Youden Index   {:>1.5f}\n"
                  "\n").format(
                               self._dict_metrics['tp']['global_value'],
                               self._dict_metrics['fp']['global_value'],
                               self._dict_metrics['ppv']['global_value'],
                               self._dict_metrics['lift_p']
                               ['global_value'],
                               self._dict_metrics['tpr']['global_value'],
                               self._dict_metrics['fpr']['global_value'],
                               self._dict_metrics['fn']['global_value'],
                               self._dict_metrics['tn']['global_value'],
                               self._dict_metrics['npv']['global_value'],
                               self._dict_metrics['lift_n']
                               ['global_value'],
                               self._dict_metrics['fnr']['global_value'],
                               self._dict_metrics['tnr']['global_value'],
                               self._dict_metrics['accuracy']
                               ['global_value'],
                               self._dict_metrics['youden']['global_value']
                               )

        report += (
                   "     Gini                   {:>1.5f} +/- {:>1.5f} (95%); std: {:>1.7f}\n"
                   "     Matthews Corr. Coef.   {:>1.5f} +/- {:>1.5f} (95%); std: {:>1.7f}\n"
                   "     Cohen Kappa            {:>1.5f} +/- {:>1.5f} (95%); std: {:>1.7f}\n"
                   "\n").format(
                                self._dict_metrics['gini']['global_value'],
                                self._dict_metrics['gini']
                                ['std_rand'] * 1.96,
                                self._dict_metrics['gini']['std_rand'],
                                self._dict_metrics['mcc']['global_value'],
                                self._dict_metrics['mcc']
                                ['std_rand'] * 1.96,
                                self._dict_metrics['mcc']['std_rand'],
                                self._dict_metrics['kappa']
                                ['global_value'],
                                self._dict_metrics['kappa']
                                ['std_rand'] * 1.96,
                                self._dict_metrics['kappa']['std_rand']
                                )

        report += (
                   "     Log-loss       {:>7.5f}\n"
                   "   \033[94m--------------------------------------------------------------------------\033[0m\n").format(self._dict_metrics['log_loss']['global_value'])

        print(report)

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
        y_prob = np.array([elt[1] for elt in self._model.predict_proba(X)])

        # Metrics derivation
        calc_metrics = binary_classification_report(y_true=y, y_pred=y_pred,
                                                    y_pred_proba=y_prob,
                                                    metrics=list(_METRICS.
                                                                 keys()))

        for name, value in calc_metrics.items():
            self._dict_metrics[name]['global_value'] = value

        # Curves
        self._roc = roc_curve(y, y_prob)[0:2]  # fpr, tpr

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
                y_prob_rand = y_prob[idx]
                y_test_rand = y[idx]

                sim_result = binary_classification_report(
                    y_true=y_test_rand,
                    y_pred=y_pred_rand,
                    y_pred_proba=y_prob_rand,
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
            self._temporal_analysis(y, y_pred, y_prob, dates)

            if self.verbose:
                print("   Time analysis completed!")

        self._time_run_metrics = time.perf_counter() - time_init

    def _temporal_analysis(self, y, y_pred, y_prob, dates):
        """Temporal analysis of model performance.

        It calculates the performance metrics of the model for 1-year windows.

        Parameters
        ----------
        y : array-like, shape = [n_samples]
            The target values.

        y_pred : array-like, shape = [n_samples]
            The target values predicted by the model.

        y_prob : array-like, shape = [n_samples]
            Score probabilities to be in each class.

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
            y_prob_rand = y_prob[ind]

            if len(y_ind) != 0 and (sum(y_ind) != 0) and \
               (sum(y_pred_ind) != 0) and (len(y_ind)-sum(y_ind) != 0):
                aux_metrics = binary_classification_report(
                    y_true=y_ind, y_pred=y_pred_ind, y_pred_proba=y_prob_rand,
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
