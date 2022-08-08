"""
Hyperopt and RBFOpt solver support.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import os
import platform
import time

import numpy as np

from scipy import stats

from ..._thirdparty.hyperopt import fmin
from ..._thirdparty.hyperopt import rand
from ..._thirdparty.hyperopt import space_eval
from ..._thirdparty.hyperopt import STATUS_FAIL
from ..._thirdparty.hyperopt import STATUS_OK
from ..._thirdparty.hyperopt import tpe
from ..._thirdparty.hyperopt import Trials

try:
    from ..._thirdparty.rbfopt import RbfoptAlgorithm
    from ..._thirdparty.rbfopt import RbfoptUserBlackBox
    from ..._thirdparty.rbfopt import RbfoptSettings

    # AMPL solvers
    system_os = platform.system()

    if system_os == "Linux":
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                   '../..', '_thirdparty'))

        MINLP_SOLVER_PATH = BASE_DIR + "/ampl_solvers/linux/bonmin"
        NLP_SOLVER_PATH = BASE_DIR + "/ampl_solvers/linux/ipopt"
    elif system_os == "Windows":
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                   '..\\..', '_thirdparty'))

        MINLP_SOLVER_PATH = BASE_DIR + "\\ampl_solvers\\windows\\bonmin.exe"
        NLP_SOLVER_PATH = BASE_DIR + "\\ampl_solvers\\windows\\ipopt.exe"
    else:
        raise RuntimeError("Unexpected system " + system_os + ".")

    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False


def _print_solver_header():
    """Print solver header iteration information."""
    header = (
        "=================================================\n"
        "  iter        score       time    best    gain\n"
        "=================================================")
    print(header)


def _print_solver_iteration(iter, score, time, best, improvement):
    """Print solver iteration information."""
    best_symbol = "B*" if best else ""
    output = "{:>6}   {:>10.5f}   {:>7.2f}s   {:>4}   {:>6.2%}".format(
        iter, score, time, best_symbol, improvement)
    print(output)


class HyperoptSolver(object):
    """Class to adapt Hyperopt solver."""
    def __init__(self, model, parameters, scorer, cv, algorithm="tpe",
                 n_iters=100, n_splits=5, random_state=None,
                 info_reporting=True, verbose=False):

        self.model = model
        self.parameters = parameters
        self.scorer = scorer
        self.cv = cv
        self.algorithm = algorithm
        self.n_iters = n_iters
        self.n_splits = n_splits
        self.random_state = random_state
        self.info_reporting = info_reporting
        self.verbose = verbose

        # timing
        self._time_solver_init = None

        # iteration information (keep track)
        self._best_score = 0
        self._best_iter = 0
        self._iter = 0

        # auxiliary data to bypass hyperopt limitation to pass more than one
        # argument to objective function
        self.X = None
        self.y = None

    def fit(self, X, y):
        """Run solver."""
        self.X = X
        self.y = y

        self._time_solver_init = time.perf_counter()

        if self.algorithm == "tpe":
            algorithm = tpe.suggest
        elif self.algorithm == "random":
            algorithm = rand.suggest

        # trials to track iterations
        trials = Trials()

        best = fmin(fn=self._objective, space=self.parameters._parameters,
                    algo=algorithm, trials=trials, max_evals=self.n_iters,
                    rstate=np.random.RandomState(self.random_state),
                    verbose=self.verbose)

        self._best_model_params = space_eval(self.parameters._parameters, best)
        self._best_score = abs(trials.best_trial['result']['loss'])

        return trials

    def _objective(self, params):
        """Objective function to minimize."""

        # set new parameters to the model
        self.model.set_params(**params)

        # cross-validation loop
        scores = []
        for train_index, test_index in self.cv.split(self.X, self.y):
            X_train, y_train = self.X[train_index], self.y[train_index]
            X_test, y_test = self.X[test_index], self.y[test_index]

            # fit new model. Return fail iteration is exception is triggered.
            try:
                self.model.fit(X_train, y_train)
                status = STATUS_OK
            except Exception as e:
                status = STATUS_FAIL
                print(str(e))

            # set new model to the scorer and compute score
            scores.append(self.scorer.score(self.model, X_test, y_test))

        # compute cross-validation scores statistics
        score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        std_score = np.std(scores)

        # Fit model with total data
        if self.info_reporting:
            # extract feature names fit feature selection
            self.model.feature_selection.fit(self.X, self.y)
            if (self.model.feature_names is not None
                    or self.model.feature_selection.feature_names is not None):
                feat_names = self.model.get_support(output_names=True)
            else:
                feat_names = self.model.get_support(indices=True)

            # extract feature coefficients
            if hasattr(self.model.estimator, "coef_"):
                self.model.estimator.fit(
                    self.model.feature_selection.transform(self.X), self.y)
                feat_coefs = self.model.estimator.coef_.ravel()
            elif hasattr(self.model.estimator, "feature_importances_"):
                self.model.estimator.fit(
                    self.model.feature_selection.transform(self.X), self.y)
                feat_coefs = self.model.estimator.feature_importances_
            else:
                feat_coefs = None
        else:
            feat_names = None
            feat_coefs = None

        # keep track of the best score
        if self._iter == 0:
            self._best_score = score

        if score > self._best_score and self._iter > 0:
            improvement = score / self._best_score - 1.0
            self._best_score = score
            self._best_iter = self._iter
        else:
            improvement = 0

        if self.verbose:
            self._print_iteration(abs(score), improvement)

        # update iteration counter
        self._iter += 1

        # build dictionary and pass to solver
        info_iter = {
            'loss': -score,
            'status': status,
            'params': params,
            'score_mean': score,
            'score_std': std_score,
            'score_min': min_score,
            'score_max': max_score,
            'feature_names': feat_names,
            'feature_coef': feat_coefs
        }

        return info_iter

    def _print_iteration(self, score, improvement):
        """Print iteration header and iteration information."""
        elapsed = time.perf_counter() - self._time_solver_init

        if self._iter % 20 == 0:
            _print_solver_header()

        _print_solver_iteration(self._iter, score, elapsed,
                                (improvement != 0) or self._iter == 0,
                                improvement)

    @property
    def best_model(self):
        return self.model.set_params(**self._best_model_params)

    @property
    def best_model_params(self):
        return self._best_model_params

    @property
    def best_model_score(self):
        return self._best_score


class DFOTrials(object):
    """Database interface supporting data-driven model-based optimization."""
    def __init__(self, parameters):
        self._parameters = parameters
        self._results = []

    def losses(self):
        # return list of losses
        return [t["loss"] for t in self._results]

    @property
    def results(self):
        # return list of dictionaries (iterations)
        return self._results

    @property
    def vals(self):
        # return dictionary with parameters
        dict_vals = {param: [] for param in self._parameters}

        for trial in self._results:
            for param, value in trial["params"].items():
                dict_vals[param].append(value)

        return dict_vals


class DFOSolver(object):
    """Class to adapt RBFOpt to HyperOpt structure."""
    def __init__(self, model, parameters, scorer, cv, n_iters=100, n_splits=5,
                 random_state=None, info_reporting=True, verbose=False):
        self.model = model
        self.parameters = parameters
        self.scorer = scorer
        self.cv = cv
        self.n_iters = n_iters
        self.n_splits = n_splits
        self.random_state = random_state
        self.info_reporting = info_reporting
        self.verbose = verbose

        # iteration information (keep track)
        self._best_score = 0
        self._best_iter = 0
        self._iter = 0

        # iteration information
        self._trials = None

        if not PYOMO_AVAILABLE:
            raise ImportError("Pyomo library is not installed. Use algorithm "
                              "'tpe' or 'random' instead.")

    def fit(self, X, y):
        """Run solver."""
        self.X = X
        self.y = y

        # trials to track iterations
        self._trials = DFOTrials(self.parameters._dict_params.keys())

        settings = RbfoptSettings(minlp_solver_path=MINLP_SOLVER_PATH,
                                  nlp_solver_path=NLP_SOLVER_PATH,
                                  max_evaluations=self.n_iters,
                                  rand_seed=self.random_state)

        lb_cons, ub_cons, type_cons = self._bound_constraints()
        n_cons = len(lb_cons)

        bb = RbfoptUserBlackBox(n_cons, lb_cons, ub_cons, type_cons,
                                self._objective)

        best = RbfoptAlgorithm(settings, bb)

        best_score, best_iteration, _, _, _ = best.optimize()

        self._best_score = abs(best_score)
        self._best_model_params = self._space_eval(best_iteration)

        return self._trials

    def _objective(self, x):
        """Objective function to minimize."""

        # build dictionary and pass to solver
        params = self._space_eval(x)

        # set new parameters to the model
        self.model.set_params(**params)

        # cross-validation loop
        scores = []
        for train_index, test_index in self.cv.split(self.X, self.y):
            X_train, y_train = self.X[train_index], self.y[train_index]
            X_test, y_test = self.X[test_index], self.y[test_index]

            # fit new model. Return fail iteration is exception is triggered.
            try:
                self.model.fit(X_train, y_train)
                status = STATUS_OK
            except Exception as e:
                status = STATUS_FAIL
                print(str(e))

            # set new model to the scorer and compute score
            scores.append(self.scorer.score(self.model, X_test, y_test))

        # compute cross-validation scores statistics
        score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        std_score = np.std(scores)

        # Fit model with total data
        if self.info_reporting:
            # extract feature names fit feature selection
            self.model.feature_selection.fit(self.X, self.y)
            if (self.model.feature_names is not None
                    or self.model.feature_selection.feature_names is not None):
                feat_names = self.model.get_support(output_names=True)
            else:
                feat_names = self.model.get_support(indices=True)

            # extract feature coefficients
            if hasattr(self.model.estimator, "coef_"):
                self.model.estimator.fit(
                    self.model.feature_selection.transform(self.X), self.y)
                feat_coefs = self.model.estimator.coef_.ravel()
            elif hasattr(self.model.estimator, "feature_importances_"):
                self.model.estimator.fit(
                    self.model.feature_selection.transform(self.X), self.y)
                feat_coefs = self.model.estimator.feature_importances_
            else:
                feat_coefs = None
        else:
            feat_names = None
            feat_coefs = None

        # keep track of the best score
        if self._iter == 0:
            self._best_score = score

        if score > self._best_score and self._iter > 0:
            self._best_score = score
            self._best_iter = self._iter

        # update iteration counter
        self._iter += 1

        # build dictionary and pass to database
        info_iter = {
            'loss': -score,
            'status': status,
            'params': params,
            'score_mean': score,
            'score_std': std_score,
            'score_min': min_score,
            'score_max': max_score,
            'feature_names': feat_names,
            'feature_coef': feat_coefs
        }

        self._trials.results.append(info_iter)

        return -score

    def _bound_constraints(self):
        """"
        Convert Parameters class to bound constraints. Return lower and upper
        bounds, and variable type.
        """
        lbs = []
        ubs = []
        var_types = []

        for param in self.parameters._dict_params.values():
            if param[0] == "uniform":
                lb, ub = param[1:]
                var_type = 'R'
            elif param[0] == "lognormal":
                mu, sigma = param[1:]
                lb, ub = stats.lognorm(loc=mu, s=sigma).ppf([0.025, 0.975])
                var_type = 'R'
            elif param[0] == "normal":
                mu, sigma = param[1:]
                lb, ub = stats.norm(loc=mu, scale=sigma).ppf([0.025, 0.975])
                var_type = 'R'
            elif param[0] == "choice":
                lb, ub = 0, len(param[1]) - 1
                var_type = 'I'
            elif param[0] == "randint":
                lb, ub = 0, param[1]
                var_type = 'I'

            lbs.append(lb)
            ubs.append(ub)
            var_types.append(var_type)

        return lbs, ubs, var_types

    def _space_eval(self, x):
        """Map solver variables to evaluation space."""
        params = {}

        for i, (param, vals) in enumerate(
                self.parameters._dict_params.items()):
            if vals[0] == "choice":
                params[param] = vals[1][int(x[i])]
            else:
                params[param] = x[i]

        return params

    @property
    def best_model(self):
        return self.model.set_params(**self._best_model_params)

    @property
    def best_model_params(self):
        return self._best_model_params

    @property
    def best_model_score(self):
        return self._best_score
