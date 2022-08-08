"""
Plot ModelOptimizer history and parameters distribution.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import matplotlib.pyplot as plt
import numpy as np

from ...core.exceptions import NotFittedException
from .optimization import ModelOptimizer


def _check_modeloptimizer(modeloptimizer):
    if not isinstance(modeloptimizer, ModelOptimizer):
        raise TypeError("modeloptimizer {} is not an instance of "
                        "ModelOptimizer".format(
                            modeloptimizer.__class__.__name__))

    if not modeloptimizer._is_fitted:
        raise NotFittedException(modeloptimizer)


def _check_modeloptimizer_parameter(modeloptimizer, parameter):
    parameters = list(modeloptimizer.parameters._parameters.keys())
    if parameter not in parameters:
        raise ValueError("parameter {} was not optimized. Optimized "
                         "parameters are {}").format(parameter, parameters)


def plot_modeloptimizer_history(modeloptimizer, parameter=None):
    """
    Plot model optimizer score or parameter value history.

    This plot shows the score or parameter value at each iteration, being
    a visual tool to assess the convergence of the algorithm to the optimal
    parameter's value.

    Parameters
    ----------
    modeloptimizer : object
        A ``ModelOptimizer`` class instance. The modeloptimizer must have been
        fitted, otherwise an exception is raised.

    parameter: str or None (default=None)
        A parameter used to optimize the model.

    See also
    --------
    grmlab.modelling.ModelOptimizer

    Examples
    --------
    >>> from grmlab.modelling.model_optimization import plot_modeloptimizer_history
    >>> plot_modeloptimizer_history(modeloptimizer)
    >>> plot_modeloptimizer_history(modeloptimizer, "estimator__C")
    """
    _check_modeloptimizer(modeloptimizer)

    n_trials = modeloptimizer.n_iters
    best_iteration = modeloptimizer._best_iter
    is_cat = False

    if parameter is not None:
        distribution = modeloptimizer.parameters._dict_params[parameter][0]
        parameter_values = modeloptimizer._trials.vals[parameter]
        best_value = modeloptimizer.best_model_parameters[parameter]
        if (distribution is 'choice' and
                modeloptimizer.algorithm in ("tpe", "random")):
            values = modeloptimizer.parameters._dict_params[parameter][1]
            parameter_values = [values[i] for i in parameter_values]

        if any(isinstance(i, str) for i in parameter_values):
            is_cat = True
            parameter_values = np.asarray(parameter_values).astype(str)
            uniques, parameter_values = np.unique(parameter_values,
                                                  return_inverse=True)

            best_value_cat = str(best_value)
            best_value = list(uniques).index(best_value_cat)

        fig, ax = plt.subplots()

        ax.scatter(range(n_trials), parameter_values)
        ax.scatter(best_iteration, best_value, marker="*", color="y", s=200,
                   label="best value = {}".format(
                    best_value_cat if is_cat else best_value))
        ax.set_ylabel(parameter)

        if is_cat:
            ax.set(yticks=range(len(uniques)), yticklabels=uniques)

        plt.legend()
    else:
        # extract scores
        scores = -np.asarray(modeloptimizer._trials.losses())

        max_score = scores[0]
        max_scores = [max_score]
        for i in range(1, n_trials):
            if scores[i] > max_score:
                max_score = scores[i]
            max_scores.append(max_score)

        s_max = [t["score_max"] for t in modeloptimizer._trials.results]
        s_min = [t["score_min"] for t in modeloptimizer._trials.results]
        s_mean = [t["score_mean"] for t in modeloptimizer._trials.results]
        # s_std = [t["score_std"] for t in modeloptimizer._trials.results]

        best_score = max_scores[-1]

        fig, axs = plt.subplots(2, 1)

        axs[0].scatter(range(n_trials), scores)
        axs[0].plot(max_scores, color="g", label="maximum score")
        axs[0].scatter(np.argmax(scores), best_score, marker="*", color="y",
                       s=200, label="best score = {:.6f}".format(best_score))
        axs[0].set_ylabel("score")
        axs[0].legend()

        axs[1].plot(range(n_trials), s_max, label="max score", color="grey",
                    linestyle="--")
        axs[1].plot(range(n_trials), s_min, label="min score", color="grey",
                    linestyle="--")
        axs[1].plot(range(n_trials), s_mean, label="mean score")
        axs[1].scatter(np.argmax(scores), best_score, marker="*", color="y",
                       s=200, label="best score = {:.6f}".format(best_score))
        axs[1].set_ylabel("score")

        axs[1].fill_between(range(n_trials), s_max, s_min, alpha=0.2)
        axs[1].legend()

    plt.xlabel("iteration")
    plt.show()


def plot_modeloptimizer_parameter(modeloptimizer, parameter):
    """
    Plot model parameter sampling throughout optimization.

    This plot shows the relationship between the score and the parameter
    distribution.

    Parameters
    ----------
    modeloptimizer : object
        A ``ModelOptimizer`` class instance. The modeloptimizer must have been
        fitted, otherwise an exception is raised.

    parameter : str
        A parameter used to optimize the model.

    See also
    --------
    grmlab.modelling.ModelOptimizer

    Examples
    --------
    >>> from grmlab.modelling.model_optimization import plot_modeloptimizer_parameter
    >>> plot_modeloptimizer_parameter(modeloptimizer, "estimator__C")
    """
    _check_modeloptimizer(modeloptimizer)
    _check_modeloptimizer_parameter(modeloptimizer, parameter)

    is_cat = False

    # extract scores
    scores = -np.asarray(modeloptimizer._trials.losses())

    # extract parameter values through optimization process
    parameter_values = modeloptimizer._trials.vals[parameter]
    best_value = modeloptimizer.best_model_parameters[parameter]

    distribution = modeloptimizer.parameters._dict_params[parameter][0]
    if (distribution is 'choice' and
            modeloptimizer.algorithm in ("tpe", "random")):
        values = modeloptimizer.parameters._dict_params[parameter][1]
        parameter_values = [values[i] for i in parameter_values]

    if any(isinstance(i, str) for i in parameter_values):
        is_cat = True
        parameter_values = np.asarray(parameter_values).astype(str)
        uniques, parameter_values = np.unique(parameter_values,
                                              return_inverse=True)

        best_value_cat = str(best_value)
        best_value = list(uniques).index(best_value_cat)

    if is_cat:
        best_label = "best value = {}".format(best_value_cat)
    else:
        best_label = "best value = {:.8f}".format(best_value)

    fig, ax = plt.subplots()

    plt.axvline(x=best_value, linestyle="--", color="y", label=best_label)
    ax.scatter(parameter_values, scores)
    ax.set_xlabel(parameter)
    ax.set_ylabel("score")

    if is_cat:
        ax.set(xticks=range(len(uniques)), xticklabels=uniques)

    plt.legend()
    plt.show()
