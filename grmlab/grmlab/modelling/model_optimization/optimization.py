"""
Hyperparameter optimization methods.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import time

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target

from ...core.base import GRMlabBase
from ...core.exceptions import NotFittedException
from ..base import GRMlabModel
from .scoring import Scorer
from .solvers import DFOSolver
from .solvers import HyperoptSolver
from .util import ModelOptimizerParameters


def _check_input(model, parameters, scorer):
    """
    Check ModelOptimizer class inputs.

    Parameters
    ----------
    model : object
        A GRMlabModel class. Otherwise, an exception is raised.

    parameters : object
        A ModelOptimizerParameters class. Otherwise, an exception is raised.

    scorer : object
        A Scorer class. Otherwise, an exception is raised.
    """
    if not isinstance(model, GRMlabModel):
        raise TypeError("model {} does not have GRMlabModel as a parent "
                        "class.".format(model.__class__.__name__))

    if not isinstance(scorer, Scorer):
        raise TypeError("scorer {} does not have Scorer as parent "
                        "class".format(scorer.__class__.__name__))

    if not isinstance(parameters, ModelOptimizerParameters):
        raise TypeError("parameters {} is not a ModelOptimizerParameters "
                        "object".format(parameters.__class__.__name__))


class ModelOptimizer(GRMlabBase):
    """
    Optimize model hyperparameters.

    Hyperparameter optimization :cite:`Bergstra_2012,Bergstra_2013` is an
    ubiquitous step in many automatic tools for Machine Learning. This process
    aims to automate the search of the optimal value for the hyperparameters of
    the feature selection method and estimator simultaneously. The goal is to
    obtain the best possible model (to be defined) under certain constraints.
    The best model: model with the highest/lowest scoring function.

    Parameters
    ----------
    model : object
        A model class instance having ``GRMlabModel`` as a parent class.

    parameters : object
        The parameters to be optimized throughout the process. This must be an
        instance of ``ModelOptimizerParameters``.

    scorer : object
        The scorer to be used as objective function to be optimized. This must
        be a scorer class instance having ``Scorer`` as a parent class.

    algorithm : str, 'dfo', random' or 'tpe' (default='dfo')
        The algorithm to investigative the variables' search space. DFO solves
        a derivative-free optimization problem adjusting bound constraints, it
        is the recommended option. TPE uses a Bayesian optimization approach.
        Random uses a simple randomized search. All three approaches differ
        from a typically employed grid search.

    n_iters : int (default=100)
        The number of different parameter settings that are sampled or
        iterations of the algorithm.

    n_split : int (default=5)
        Number of folds for cross-validation (CV). Must be at least 2.

    random_state : int or None (default=None)
        The seed useb by the random number generator.

    verbose : int or boolean (default=False)
        Controls verbosity of output.

    See also
    --------
    grmlab.modelling.base.GRMlabModel, Scorer, ModelOptimizerParameters

    Notes
    -----
    The current implementation uses the Bayesian optimization solver Hyperopt
    :cite:`Bergstra_2015` as a backend solver for hyperparameter optimization
    using TPE and Random algorithm. The derivative-free optimization solver
    RBFOpt :cite:`Costa_2018` is used when DFO algorithm is chosen.

    References
    ----------
    .. bibliography:: refs.bib
       :filter: docname in docnames
    """
    def __init__(self, model, parameters, scorer, algorithm="dfo", n_iters=100,
                 n_splits=5, random_state=None, info_reporting=True,
                 verbose=False):
        self.model = model
        self.parameters = parameters
        self.scorer = scorer
        self.algorithm = algorithm
        self.n_iters = n_iters
        self.n_splits = n_splits
        self.random_state = random_state
        self.info_reporting = info_reporting
        self.verbose = verbose

        # additional information
        self._target_dtype = None

        # iteration information (keep track)
        self._trials = None

        # best model results
        self._best_iter = None
        self._best_model_params = None
        self._best_model_score = None

        # timing
        self._time_solver = None

        # flag
        self._is_fitted = False

    def fit(self, X, y):
        """
        Fit model optimizer will all sets of parameters.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
        """

        # check input
        _check_input(self.model, self.parameters, self.scorer)

        # check X, y consistency
        X, y = check_X_y(X, y, "csc")

        # check target type
        self._target_dtype = type_of_target(y)

        if self._target_dtype in ("binary", "multiclass",
                                  "multilabel-indicator"):
            cv = StratifiedKFold(n_splits=self.n_splits)
        else:
            cv = KFold(n_splits=self.n_splits)

        # check algorithm
        if self.algorithm in ("random", "tpe"):
            solver = HyperoptSolver(
                model=self.model,
                parameters=self.parameters, scorer=self.scorer, cv=cv,
                algorithm=self.algorithm, n_iters=self.n_iters,
                n_splits=self.n_splits, random_state=self.random_state,
                info_reporting=self.info_reporting, verbose=self.verbose)

        elif self.algorithm == "dfo":
            solver = DFOSolver(
                model=self.model, parameters=self.parameters,
                scorer=self.scorer, cv=cv, n_iters=self.n_iters,
                n_splits=self.n_splits, random_state=self.random_state,
                info_reporting=self.info_reporting, verbose=self.verbose)
        else:
            raise ValueError("algorithm {} not supported. Available "
                             "algorithms are: dfo, random and tpe."
                             .format(self.algorithm))

        # run solver and return iteration information
        time_solver_init = time.perf_counter()

        self._trials = solver.fit(X, y)

        # retrieve simulation results
        self._best_model_score = solver.best_model_score
        self._best_model_params = solver.best_model_params
        self._best_iter = solver._best_iter

        self._time_solver = time.perf_counter() - time_solver_init

        self._is_fitted = True

        return self

    @property
    def best_model(self):
        """Return a model with best parameters."""
        if not self._is_fitted:
            raise NotFittedException(self)

        return self.model.set_params(**self._best_model_params)

    @property
    def best_model_score(self):
        """Return best model score."""
        if not self._is_fitted:
            raise NotFittedException(self)

        return self._best_model_score

    @property
    def best_model_parameters(self):
        """Return best model parameters."""
        if not self._is_fitted:
            raise NotFittedException(self)

        return self._best_model_params
