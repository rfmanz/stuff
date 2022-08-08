"""
Model Comparison
"""

# Authors: Carlos González Berrendero <c.gonzalez.berrender@bbva.com
# BBVA - Copyright 2019.

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from ...core.base import GRMlabBase
from ...core.exceptions import NotFittedException
from ...core.exceptions import NotRunException
from ..base import GRMlabModel
from ..classification import GRMlabModelClassification
from ..feature_selection import SelectOptimize
from ..model_optimization import ModelOptimizer
from .analyzer import ModelAnalyzer
from .plots import plot_roc_comp


_NUM_MAX_VARS_MODEL_DIFF = 20
_NUM_MIN_VARS_MODEL_DIFF = 4

_MAX_CORRELATION_MODEL_DIFF = 1

_METRICS_STATS = ['gini', 'kappa', 'mcc', 'ppv', 'lift_p',
                  'npv', 'lift_n', 'accuracy',
                  'balanced_accuracy', 'youden', 'log_loss']

_METRIC_RAND_COMPUTED = ['gini', 'kappa', 'mcc']

_INTERVALS_STATS = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

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


def str_dif_bar(len_max, value_1, value_2, interval, std_rand=0):
    """
    Generate a bar for difference report in the form:
    ---------- ----------
                  |       difference  value_1 value_2
    ---------- ----------
    Parameters
    ----------
    len_max: max length of variable name.
    value_1: metric in model 1.
    value_2: metric in model 2.
    interval: max value shown in the bar.
    std_rand: 1 to show confidence intervals (gini, mcc and kappa)

    Returns
    -------
    string with 3 lines to add to the report.
    """
    previous_mark = int((value_2-value_1)/interval * 10 + 10)
    previous_mark = max(previous_mark, 0)
    previous_mark = min(previous_mark, 20)
    string = "  "+" "*len_max + "    ---------- ----------  \n"
    if std_rand == 0:
        string += "  {:>" + str(len_max) + "}    " + " "*previous_mark + "|" \
               + " " * (21 - previous_mark)  \
               + " {:+5.2f}     {:4.4f}                {:4.4f}\n"
    else:
        string += "  {:>" + str(len_max) + "}    " + " " * previous_mark \
               + "|" + " " * (21 - previous_mark)  \
               + " {:+5.2f}     {:4.4f}(+/-{:4.4f})     {:4.4f}(+/-{:4.4f})\n"
    return string


class ModelComparison(GRMlabBase):
    """It compares two fitted models over a given dataset.
    It compares the performance metrics, features and  predictions of
    two different models.

    Parameters
    ----------
    name : str
        Name given to the model comparison.

    model1 : :class:`GRMlabModel`, :class:`GRMlabModelClassification`
        The first fitted model to be compared.

    model2 : :class:`GRMlabModel`, :class:`GRMlabModelClassification`
        The second fitted model to be compared.

    num_simulations : int or None (default=100)
        Indicates the number of simulations to be executed. It has to be
        positive. If None, no simulations will be done.

    simulation_size : float (default=0.5)
        Size of the simulation data relative to number of elements in X.
        The value has to be in the interval (0, 1).

    feature_analysis_approximation : boolean (default=False)
        If True, it will calculate only first order approximations for
        non-analytical calculations.

    feature_names: needed when the models has been created with this
        feature_names.

    verbose : boolean (default=True)
        If True, it will print verbose during the execution.
    """
    def __init__(self, name, model1=None, model2=None, n_simulations=100,
                 simulation_size=0.5, seed=None,
                 feature_analysis_approximation=False, feature_names=None,
                 verbose=True):

        # models variables
        self._model1 = model1
        self._model2 = model2

        self.name = name
        self.n_simulations = n_simulations
        self.simulation_size = simulation_size
        self.rand_seed = seed
        self.feature_analysis_approximation = feature_analysis_approximation
        self.feature_names = feature_names
        self.verbose = verbose

        # Analyzers
        self._analyzer1 = None
        self._analyzer2 = None

        # Comparison info
        self._common_features = []
        self._only_in_1_features = []
        self._only_in_2_features = []

        self._predictions = [[], []]
        self._disagreement_predictions_sum = []
        self._agreement_predictions_sum = []

        self._model_difference = None
        self._analyzer_difference = None
        self._is_there_difference = False

        self._l2_dif_pred = None
        self._l2_dif_feature = None

        # Flags
        self._is_dates_provided = False
        self._is_run = False

    def run(self, X, y, dates=None, feature_names=None):
        """
        Runs model analysis over the two models and generates the
        difference model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
                The test input samples.

        y : array-like, shape = [n_samples]
            The test target values.

        dates : array-like, shape = [n_samples] or [n_dates] (default=None)
            Dates on the test data. This parameter is needed for time analysis.
        """
        if dates is not None:
            self._is_dates_provided = True
        else:
            self._is_dates_provided = False
        # check models
        if isinstance(self._model1, (GRMlabModelClassification, GRMlabModel)):
            if not self._model1._is_estimator_fitted:
                raise NotFittedException(self._model1)
        elif isinstance(self._model1, ModelOptimizer):
            self._model1 = self._model1.best_model
        else:
            raise TypeError("class {} is not an accepted model class.".
                            format(type(self._model1).__name__))
        if isinstance(self._model2, (GRMlabModelClassification, GRMlabModel)):
            if not self._model2._is_estimator_fitted:
                raise NotFittedException(self._model2)
        elif isinstance(self._model2, ModelOptimizer):
            self._model2 = self._model2.best_model
        else:
            raise TypeError("class {} is not an accepted model class.".
                            format(type(self._model2).__name__))

        # Run analyzers
        feature_approx = self.feature_analysis_approximation
        self._analyzer1 = ModelAnalyzer(
            "model1", self._model1, n_simulations=self.n_simulations,
            simulation_size=self.simulation_size, seed=self.rand_seed,
            feature_analysis_approximation=feature_approx, verbose=self.verbose
        )
        self._analyzer2 = ModelAnalyzer(
            "model2", self._model2, n_simulations=self.n_simulations,
            simulation_size=self.simulation_size, seed=self.rand_seed,
            feature_analysis_approximation=feature_approx, verbose=self.verbose
        )

        self._analyzer1.run(X, y, dates)
        self._analyzer2.run(X, y, dates)

        self._feature_comparison()
        self._predict_comparison(X, y)

        self._is_run = True

    def _feature_comparison(self):
        """
        Comparison between the features of each model.
            Generate list of common features and L2 distance.
        """

        self._common_features = []
        self._only_in_1_features = []
        self._only_in_2_features = []

        for var1 in self._analyzer1._feature_importance:
            for var2 in self._analyzer2._feature_importance:
                if var1[0] == var2[0]:
                    self._common_features.append(var1 + (var2[1],))
        self._only_in_1_features = [
                var for var in self._analyzer1._feature_importance
                if var[0] not in self._analyzer2._feature_names]
        self._only_in_2_features = [
                var for var in self._analyzer2._feature_importance
                if var[0] not in self._analyzer1._feature_names]

        dif = sum((v[1]-v[2]) ** 2 for v in self._common_features)
        dif += sum(v[1] ** 2 for v in self._only_in_1_features)
        dif += sum(v[1] ** 2 for v in self._only_in_2_features)

        count_variables = len(self._common_features)
        + len(self._only_in_1_features)
        + len(self._only_in_2_features)
        self._l2_dif_feature = dif / count_variables

    def _predict_comparison(self, X, y):
        """
        Comparison between predictions over set X.
        """
        self._predictions = [[], []]
        self._disagreement_predictions_sum = []
        self._agreement_predictions_sum = []

        self._predictions[0].append(self._model1.predict(X))
        self._predictions[0].append(self._model1.predict_proba(X)[:, 1])

        self._predictions[1].append(self._model2.predict(X))
        self._predictions[1].append(self._model2.predict_proba(X)[:, 1])

        square_difference = [d**2 for d in self._predictions[0][1] -
                             self._predictions[1][1]]

        self._l2_dif_pred = sum(square_difference)/len(square_difference)

        #    y  Mod1    Mod2
        #    1    1       0
        #    1    0       1
        #
        #    0    1       0
        #    0    0       1
        logic_dis_aux_0 = y & self._predictions[0][0] \
            & (~ self._predictions[1][0])
        self._disagreement_predictions_sum.append(sum(logic_dis_aux_0))

        logic_dis_aux_1 = y & (~ self._predictions[0][0]) \
            & self._predictions[1][0]
        self._disagreement_predictions_sum.append(sum(logic_dis_aux_1))

        logic_dis_aux_2 = (~ y) & self._predictions[0][0] \
            & (~ self._predictions[1][0])
        self._disagreement_predictions_sum.append(sum(logic_dis_aux_2))

        logic_dis_aux_3 = (~ y) & (~ self._predictions[0][0]) \
            & self._predictions[1][0]
        self._disagreement_predictions_sum.append(sum(logic_dis_aux_3))

        #    y  Mod1    Mod2
        #    1    1       1
        #    1    0       0
        #
        #    0    1       1
        #    0    0       0
        logic_aux = (y.astype(bool) & self._predictions[0][0].astype(bool)
                     & self._predictions[1][0].astype(bool))
        self._agreement_predictions_sum.append(sum(logic_aux))

        logic_aux = (y.astype(bool) & (~ self._predictions[0][0].astype(bool))
                     & (~ self._predictions[1][0].astype(bool)))
        self._agreement_predictions_sum.append(sum(logic_aux))

        logic_aux = ((~ y.astype(bool)) & self._predictions[0][0].astype(bool)
                     & self._predictions[1][0].astype(bool))
        self._agreement_predictions_sum.append(sum(logic_aux))

        logic_aux = ((~ y.astype(bool))
                     & (~ self._predictions[0][0].astype(bool))
                     & (~ self._predictions[1][0].astype(bool)))
        self._agreement_predictions_sum.append(sum(logic_aux))

        # Generate model difference.
        # Generate a random forest model that tries to guess whether the
        # two models diverge in their responses or not. It can be useful
        # to determine segmentation where apply different models.

        model_1_correct = logic_dis_aux_0 | logic_dis_aux_3
        model_2_correct = logic_dis_aux_1 | logic_dis_aux_2

        complete_disagreement = (model_1_correct | model_2_correct).astype(int)

        X_sub = X[[x == 1 for x in complete_disagreement], :]
        target_difference = model_1_correct[[x == 1
                                             for x in complete_disagreement]]
        target_difference = np.array([int(x) for x in target_difference])

        optimize = SelectOptimize(method="classification",
                                  n_max_features=_NUM_MAX_VARS_MODEL_DIFF,
                                  n_min_features=_NUM_MIN_VARS_MODEL_DIFF,
                                  max_correlation=_MAX_CORRELATION_MODEL_DIFF)
        rand_forest_model = RandomForestClassifier(class_weight="balanced")
        model_difference_not_fitted = GRMlabModelClassification(
            name="Difference", feature_names=self.feature_names,
            feature_selection=optimize, estimator=rand_forest_model)

        if sum(target_difference) == 0:
            if self.verbose:
                print("Models are not different enough to train" +
                      " a difference model")
        else:
            self._model_difference = model_difference_not_fitted.fit(
                X_sub, target_difference)

            self._analyzer_difference = ModelAnalyzer("Difference",
                                                      self._model_difference,
                                                      verbose=False)
            self._analyzer_difference.run(X_sub, target_difference)
            self._is_there_difference = True

    def stats(self, step="metrics"):
        """Model comparison results.

        Parameters
        ----------
        step : str (default="metrics")
            Options are "prediction", "metrics", and "features".
        """
        if not self._is_run:
            raise NotRunException(self, "run")

        if step not in ("metrics", "features", "prediction"):
            raise ValueError("step not found.")

        if step == "prediction":
            agree_correct_pred = (self._agreement_predictions_sum[0] +
                                  self._agreement_predictions_sum[3])
            agree_incorrect_pred = (self._agreement_predictions_sum[1] +
                                    self._agreement_predictions_sum[2])
            agree_pred = agree_correct_pred + agree_incorrect_pred

            list_format = []
            list_format.append(self._analyzer1._n_samples)
            list_format.append(agree_pred)
            list_format.append(agree_pred / self._analyzer1._n_samples * 100)
            list_format.append(agree_correct_pred)
            list_format.append(agree_correct_pred / self._analyzer1._n_samples
                               * 100)
            list_format.append(agree_incorrect_pred)
            list_format.append(agree_incorrect_pred /
                               self._analyzer1._n_samples * 100)
            list_format.append(self._l2_dif_pred)

            if self._is_there_difference:
                list_format.append(self._analyzer_difference.
                                   _feature_importance[0][0])
                list_format.append(self._analyzer_difference.
                                   _feature_importance[0][1])
                if (self._analyzer_difference._feature_importance[0][0] in
                        self._analyzer1._feature_names):
                    list_format.append('*')
                else:
                    list_format.append(' ')
                if (self._analyzer_difference._feature_importance[0][0] in
                        self._analyzer2._feature_names):
                    list_format.append('*')
                else:
                    list_format.append(' ')
            else:
                list_format += 2 * [0] + 2 * [' ']

            list_format.append(self._disagreement_predictions_sum[0])
            list_format.append(self._disagreement_predictions_sum[1])

            if self._is_there_difference:
                list_format.append(self._analyzer_difference.
                                   _feature_importance[1][0])
                list_format.append(self._analyzer_difference.
                                   _feature_importance[1][1])
                if (self._analyzer_difference._feature_importance[1][0] in
                        self._analyzer1._feature_names):
                    list_format.append('*')
                else:
                    list_format.append(' ')
                if (self._analyzer_difference._feature_importance[1][0] in
                        self._analyzer2._feature_names):
                    list_format.append('*')
                else:
                    list_format.append(' ')
                list_format.append(self._analyzer_difference.
                                   _feature_importance[2][0])
                list_format.append(self._analyzer_difference.
                                   _feature_importance[2][1])

                if (self._analyzer_difference._feature_importance[2][0] in
                        self._analyzer1._feature_names):
                    list_format.append('*')
                else:
                    list_format.append(' ')
                if (self._analyzer_difference._feature_importance[2][0] in
                        self._analyzer2._feature_names):
                    list_format.append('*')
                else:
                    list_format.append(' ')
            else:
                list_format += 2 * [0, 0, ' ', ' ']

            list_format.append(self._disagreement_predictions_sum[2])
            list_format.append(self._disagreement_predictions_sum[3])

            if self._is_there_difference:
                list_format.append(self._analyzer_difference.
                                   _feature_importance[3][0])
                list_format.append(self._analyzer_difference.
                                   _feature_importance[3][1])
                if (self._analyzer_difference._feature_importance[3][0] in
                        self._analyzer1._feature_names):
                    list_format.append('*')
                else:
                    list_format.append(' ')
                if (self._analyzer_difference._feature_importance[3][0] in
                        self._analyzer2._feature_names):
                    list_format.append('*')
                else:
                    list_format.append(' ')
            else:
                list_format += 2 * [0] + 2 * [' ']

            # Model names:
            list_format.append(self._model1.name)
            list_format.append(self._model2.name)

            if self._is_there_difference:
                len_max = max(len(var[0]) for var in
                              self._analyzer_difference.
                              _feature_importance[0:4])
            else:
                len_max = 1

            report = (
                "\033[94m================================================================================"
                + "="*(len_max - 2) + "\033[0m\n"
                "\033[1m\033[94m                         GRMlab Comparison Analysis                       \033[0m\n"
                "\033[94m================================================================================"
                + "=" * (len_max - 2) + "\033[0m\n"
                "\n"
                " Total samples: {:>10} \n"
                "      Coincident predictions: {:>10} " + "({:4.2f}%)" + "\n"
                "          Correct:   {:>10} " + "({:4.2f}%)" + "\n"
                "          Incorrect: {:>10} " + "({:4.2f}%)" + "\n"
                " L2 predict prob. distance: {:4.4f}\n"
                "\n"
                " \033[1mNon-coincident predictions                Model disagreement features: \033[0m\n"
                "        Model 1 - Model 2 predictions         Name"
                + " "*(len_max - 2) + " | Importance     M1   M2  \n"
                "  V        1 - 0    |       0 - 1          -----------------------------------"
                + "-" * (len_max - 2) + "\n"
                "  a    -----------------------------            {:<"
                + str(len_max) + "} | {:4.4f}         {:1}    {:1}\n"
                "  l 1 |  {:>10} | {:>10}                {:<" + str(len_max)
                + "} | {:4.4f}         {:1}    {:1}\n"
                "  u    -----------------------------            {:<"
                + str(len_max) + "} | {:4.4f}         {:1}    {:1}\n"
                "  e 0 |  {:>10} | {:>10}                {:<"
                + str(len_max) + "} | {:4.4f}         {:1}    {:1}\n"
                "  s   ------------------------------\n"
                "\n"
                " Model 1: {:<25}\n"
                " Model 2: {:<25}\n"
                "\n"
                "\033[94m--------------------------------------------------------------------------------"
                + "-" * (len_max - 2) + "\033[0m\n"
                "\n"
                "\n").format(*list_format)

        if step == "features":

            all_features = self._common_features.copy()
            maxs = [max(var[1], var[2]) for var in all_features]

            for var in self._only_in_1_features:
                var_append = (*var, '*')
                maxs.append(var[1])
                all_features.append(var_append)
            for var in self._only_in_2_features:
                var_append = (var[0], '*', var[1])
                maxs.append(var[1])
                all_features.append(var_append)
            ind_sort = np.argsort(maxs)
            all_features = np.array(all_features)
            all_features = all_features[ind_sort[::-1]]
            len_max = max([len(x[0]) for x in all_features])
            for var in all_features:
                var[1] = var[1][0:6]
                var[2] = var[2][0:6]

            report = (
                "\033[94m================================================================================\033[0m\n"
                "\033[1m\033[94m                         GRMlab Comparison Analysis                       \033[0m\n"
                "\033[94m================================================================================\033[0m\n"
                "\n"
                " Common variables: {:2d} \n"
                " Only in 1: {:2d} \n"
                " Only in 2: {:2d} \n"
                " L2 feature importance distance: {:4.4f}\n"
                ""
                "\n"
                " \033[1m " + " " * 51 + "Importance  \033[0m\n"
                " Feature " + " " * (len_max + 16) + "Model 1     Model 2 \n")

            list_format = []
            list_format.append(len(self._common_features))
            list_format.append(len(self._only_in_1_features))
            list_format.append(len(self._only_in_2_features))
            list_format.append(self._l2_dif_feature)

            for var in all_features:
                list_format.append(var[0])
                list_format.append(var[1])
                list_format.append(var[2])
                report += ("   {:<" + str(len_max) +
                           "}                      {:>6}      {:>6} \n")

            list_format.append(self._model1.name)
            list_format.append(self._model2.name)

            report += " \033[94m-----------------------------------------------------------------------------\033[0m\n"
            report += "  Model 1: {:<25}\n"
            report += "  Model 2: {:<25}\n"
            report += " \033[94m-----------------------------------------------------------------------------\033[0m\n\n\n"
            report = report.format(*list_format)

        if step == "metrics":

            list_format = []
            len_max = max([len(name) for name in _METRICS_STATS])

            report = (
                "\033[94m===============================================================================================\033[0m\n"
                "\033[1m\033[94m                                GRMlab Comparison Analysis                               \033[0m\n"
                "\033[94m===============================================================================================\033[0m\n"
                "\n"
                + " " * (len_max - 4) + "Metric" + "          difference                    Model 1               Model 2 \n"
                + "  "
                + " " * len_max + "              V                       \n")
            i = 0
            for name in _METRICS_STATS:
                list_format.append(name)
                value_model_1 = (self._analyzer1.
                                 _dict_metrics[name]['global_value'])
                value_model_2 = (self._analyzer2.
                                 _dict_metrics[name]['global_value'])
                list_format.append(round(value_model_2 - value_model_1, 3))
                list_format.append(value_model_1)

                # For gini, mcc and kappa.
                if (name in _METRIC_RAND_COMPUTED
                        and self.n_simulations is not None):
                    std_rand = 1
                    std_rand_model1 = (self._analyzer1.
                                       _dict_metrics[name]['std_rand'])
                    std_rand_model2 = (self._analyzer2.
                                       _dict_metrics[name]['std_rand'])
                    list_format.append(std_rand_model1 * 1.96)
                    list_format.append(value_model_2)
                    list_format.append(std_rand_model2 * 1.96)
                else:
                    list_format.append(value_model_2)
                    std_rand = 0
                if name == 'log_loss':
                    report += str_dif_bar(len_max, value_model_2,
                                          value_model_1, _INTERVALS_STATS[i],
                                          std_rand)
                else:
                    report += str_dif_bar(len_max, value_model_1,
                                          value_model_2, _INTERVALS_STATS[i],
                                          std_rand)
                i += 1

            # Model names:
            list_format.append(self._model1.name)
            list_format.append(self._model2.name)

            report += "  " + " " * len_max + "    ---------- ----------  \n"
            report += " \033[94m-----------------------------------------------------------------------------\033[0m\n"
            report += "  Model 1: {:<25}\n"
            report += "  Model 2: {:<25}\n"
            report += " \033[94m-----------------------------------------------------------------------------\033[0m\n\n\n"

            report = report.format(*list_format)

        print(report)

    def plot_roc(self):
        """Plots comparison between ROC curves.
        """
        plot_roc_comp(self._model1.name, *self._analyzer1._roc,
                      self._model2.name, *self._analyzer2._roc)

    def metric_plot_temporal(self, name, display_obs=False):
        """
        It plots the comparative temporal analysis for a metric.
        The available metrics for temporal analysis are::
            metrics = ['gini', 'default_rate', 'balanced_accuracy',
            'balanced_error_rate', 'diagnostic_odds_ratio',
            'discriminant_power', 'fnr', 'fpr', 'geometric_mean',
            'positive_likelihood', 'tpr','tnr', 'youden']

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

        def formatter_semester(x, pos=None):
            return x_axis[x] if x % 12 in (6, 12) else ""
        mt = mticker.FuncFormatter(formatter_semester)

        x_axis = [str(month[0]) for month in
                  self._analyzer1._intervals_temporal]

        y1_axis = self._analyzer1._dict_metrics[name]['temporal']
        y2_axis = self._analyzer2._dict_metrics[name]['temporal']

        ind_no_nan = np.logical_and([x is not None for x in y1_axis],
                                    [x is not None for x in y2_axis])
        y1_axis_no_nan = np.array(y1_axis)[ind_no_nan]
        y2_axis_no_nan = np.array(y2_axis)[ind_no_nan]

        fig, ax = plt.subplots()
        ax.plot(x_axis, y1_axis_no_nan, color=_GREEN_COLOR)
        ax.plot(x_axis, y2_axis_no_nan, color=_GREEN_COLOR)

        plt.title("{} time analysis".format(_METRICS[name]))
        ax.set_xlabel('Time')
        ax.set_ylabel('1-year {}'.format(_METRICS[name]))

        ax.legend(['Model 1', 'Model 2'])

        if display_obs:
            y_second_axis = [x if (x is not None) else np.nan
                             for x in self._analyzer1._count_temporal]
            second_ax = ax.twinx()
            second_ax.xaxis.set_major_formatter(mt)
            second_ax.bar(x_axis, y_second_axis, alpha=0.5, color=_BLUE_COLOR)
        for lab in ax.get_xticklabels():
            lab.set_rotation(30)

        std_metric1 = np.std(y1_axis_no_nan)
        std_metric2 = np.std(y2_axis_no_nan)
        report = (
            "\033[94m================================================================================\033[0m\n"
            "\033[1m\033[94m               GRMlab Temporal Comparison Analysis                       \033[0m\n"
            "\033[94m================================================================================\033[0m\n"
            "\n"
            " \033[1mMetric: {:>10}\033[0m\n"
            "                              Model 1               Model 2\n"
            "   Global value:              {:>0}                {:>0}\n"
            "   Range:                  {:>0}-{:>0}         {:>0}-{:>0} \n"
            "   Standard deviation:        {:>0}                {:>0}\n"
            "\033[94m-------------------------------------------------------------------------------\033[0m\n"
            "  Model 1: {:<25}\n"
            "  Model 2: {:<25}\n"
            "\033[94m-------------------------------------------------------------------------------\033[0m\n"
            ).format(_METRICS[name],
                     round(self._analyzer1._dict_metrics[name]['global_value'],
                           4),
                     round(self._analyzer2._dict_metrics[name]['global_value'],
                           4),
                     round(np.min(y1_axis_no_nan), 4),
                     round(np.max(y1_axis_no_nan), 4),
                     round(np.min(y2_axis_no_nan), 4),
                     round(np.max(y2_axis_no_nan), 4),
                     round(std_metric1, 4), round(std_metric2, 4),
                     self._model1.name,
                     self._model2.name)
        print(report)
