"""
Model generator
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import time

import pandas as pd

from joblib import delayed
from joblib import Parallel
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target

from ...core.base import GRMlabBase
from ...core.exceptions import NotRunException
from ...reporting.util import reporting_output_format
from ..classification import GRMlabModelClassification
from ..feature_selection import condition_number
from ..feature_selection import correlation_analysis
from ..regression import GRMlabModelRegression


def _check_feature_selection_estimator(feature_selection, estimator, verbose):
    """
    Check the availability of the feature selection and estimator objects.
    If only one of them is available, print info message (if verbose).
    """
    if estimator is None:
        raise ValueError("At least an estimator is required")
    # elif feature_selection is None:
    #   raise ValueError("At least a feature selection is required")


def _model_description(feature_selection, estimator):
    """
    Returns description of the model combining parameters for feature
    selection and estimator.
    """
    if feature_selection is None:
        return {"estimator": estimator.get_params()}
    else:
        return {"feature_selection": feature_selection.get_params(),
                "estimator": estimator.get_params()}


def _model_generator_results(dict_model_metrics, format):
    """Construct pandas dataframe. Optionally return dict."""
    header = list(dict_model_metrics[0].keys())
    main = ["id", "description", "feature_selection", "estimator",
            "n_features", "condition_number", "max_corr", "time_total",
            "time_feature_selection", "time_estimator"]
    rest = sorted([h for h in header if h not in main])
    new_header = main + rest
    summary = pd.DataFrame(dict_model_metrics, columns=new_header)

    return reporting_output_format(summary, format)


def _model_name(id_feature_selection, id_estimator):
    """Generate model id."""
    str_id_fsm = str(id_feature_selection)
    str_id_est = str(id_estimator)
    id_fsm = "0"*(4 - len(str_id_fsm)) + str_id_fsm
    id_est = "0"*(4 - len(str_id_est)) + str_id_est

    return "{}_{}".format(id_fsm, id_est)


def _parallel_fit_estimator(X_train, y_train, X_test, y_test,
                            feature_selection, estimator, id_fsm, id_est,
                            target_dtype, time_fsm, feature_names, verbose):
    """Private function used to fit a single model in parallel."""
    time_est_init = time.perf_counter()
    new_estimator = clone(estimator)
    new_estimator.fit(X_train, y_train)
    time_est = time.perf_counter() - time_est_init

    name = _model_name(id_fsm, id_est)
    description = _model_description(feature_selection, new_estimator)

    if target_dtype in ("binary", "multiclass"):
        model = GRMlabModelClassification(
            name=name,
            description=description, feature_names=feature_names,
            feature_selection=feature_selection, estimator=new_estimator,
            verbose=verbose)
    else:
        model = GRMlabModelRegression(
            name=name,
            description=description, feature_names=feature_names,
            feature_selection=feature_selection, estimator=new_estimator,
            verbose=verbose)

    # workaround to handle None or different feature selection methods
    if feature_selection is not None:
        model._is_feature_selection_fitted = True

    model._is_estimator_fitted = True

    if target_dtype in ("binary", "multiclass"):
        metrics = model.metrics(
            X_test, y_test, binary=(target_dtype == "binary"),
            output_dict=True)
    else:
        metrics = model.metrics(X_test, y_test, output_dict=True)

    # condition number and correlation
    if feature_selection is not None:
        X_test_transform = feature_selection.transform(X_test)
    else:
        X_test_transform = X_test

    cond = condition_number(X_test_transform)
    corr = correlation_analysis(X_test_transform)
    max_corr = max(abs(corr[0]), abs(corr[1]))

    if feature_selection is not None:
        fs_class_name = feature_selection.__class__.__name__
        fs_n_features = feature_selection.n_features_
    else:
        fs_class_name = "not set"
        fs_n_features = X_test.shape[1]

    # metrics
    metrics["id"] = name
    metrics["feature_selection"] = fs_class_name
    metrics["estimator"] = new_estimator.__class__.__name__
    metrics["description"] = description
    metrics["n_features"] = fs_n_features
    metrics["condition_number"] = cond
    metrics["max_corr"] = max_corr
    metrics["time_total"] = time_fsm + time_est
    metrics["time_feature_selection"] = time_fsm
    metrics["time_estimator"] = time_est

    return metrics, model


class ModelGenerator(GRMlabBase):
    """
    Model generator.

    Generate model candidates by combining several feature selection methods
    and estimators. Once all combinations were run, the user can select a
    specific model to perform hyperparameter optimization or further analyses.

    Parameters
    ----------
    feature_names : array, shape (n_features, ) or None (default=None)
        The name of the features.

    feature_selection : list of objects or None (default=None)
        List of feature selection algorithms with methods ``fit``,
        ``fit_transform`` and ``transform``, and attribute ``support_``.

    estimator : list of objects or None (default=None)
        List of supervised learning estimators with a ``fit`` method.

    test_size : float, int or None (default=0.3)
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size.

    random_state : int or None (default=None)
        If int, random_state is the seed used by the random number generator.
        This is used to guarantee the reproducibility of results.

    n_jobs : int (default=1)
        The number of estimators to run in parallel. Fast estimators do not
        benefit considerably due to the inherent cost of setting multiple
        processes.

    verbose : int or boolean (default=False)
        Controls verbosity of output.

    Attributes
    ----------
    n_models_ : int
        The number of models generated.
    """
    def __init__(self, feature_names=None, feature_selection=None,
                 estimator=None, test_size=0.3, random_state=None, n_jobs=1,
                 verbose=False):

        self.feature_names = feature_names
        self.feature_selection = feature_selection
        self.estimator = estimator
        self.test_size = test_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.n_models_ = None
        self._n_feature_selection = None
        self._n_estimator = None

        self._target_dtype = None
        self._n_samples = None
        self._n_features = None
        self._n_samples_train = None
        self._n_samples_test = None

        self._time_run = None

        self._models_metrics = []
        self._models_classes = []

        self._is_run = False

    def run(self, X, y):
        """
        Fit all possible combinations of feature selection methods and
        estimators. At least one estimator is required. If no feature selection
        is performed the estimators are fitted with the original training
        samples without prior transformation.

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
        time_init = time.perf_counter()

        X, y = check_X_y(X, y, "csc")

        _check_feature_selection_estimator(self.feature_selection,
                                           self.estimator, self.verbose)

        self._n_estimator = len(self.estimator)
        if self.feature_selection is not None:
            self._n_feature_selection = len(self.feature_selection)
        else:
            self._n_feature_selection = 0

        self._n_samples, self._n_features = X.shape

        self._target_dtype = type_of_target(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y)

        self._n_samples_train = X_train.shape[0]
        self._n_samples_test = X_test.shape[0]

        collect_processes = []
        if self.feature_selection is None:
            process = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose,
                prefer="processes")(
                delayed(_parallel_fit_estimator)(
                    X_train, y_train, X_test, y_test, None, estimator, 0,
                    id_est, self._target_dtype, 0, self.feature_names,
                    self.verbose
                    ) for id_est, estimator in enumerate(self.estimator))

            collect_processes.append(process)
        else:
            for id_fsm, feature_selection in enumerate(self.feature_selection):
                time_fsm_init = time.perf_counter()
                X_transform = feature_selection.fit_transform(X_train, y_train)
                time_fsm = time.perf_counter() - time_fsm_init

                process = Parallel(
                    n_jobs=self.n_jobs, verbose=self.verbose,
                    prefer="processes")(
                    delayed(_parallel_fit_estimator)(
                        X_transform, y_train, X_test, y_test,
                        feature_selection, estimator, id_fsm, id_est,
                        self._target_dtype, time_fsm,
                        self.feature_names, self.verbose
                        ) for id_est, estimator in enumerate(self.estimator))

                collect_processes.append(process)

        for process in collect_processes:
            for models_info in process:
                self._models_metrics.append(models_info[0])
                self._models_classes.append(models_info[1])

        self._is_run = True

        if self.feature_selection is None:
            self.n_models_ = len(self.estimator)
        else:
            self.n_models_ = len(self.feature_selection) * len(self.estimator)

        self._time_run = time.perf_counter() - time_init

        return self

    def get_model(self, model_id):
        """
        Get a model class instance derived from
        :class:`grmlab.modelling.base.GRMlabModel`. The returned model class is
        already fitted, therefore the user can directly call functions such as
        ``predict`` or ``predict_proba``.

        Parameters
        ----------
        model_id : str
            The id must has been generated and included in ``results`` to be
            valid.

        Example
        -------
        >>> model_generator.get_model("0000_0001")
        GRMlabModelClassification(description={"feature_selection":
        {"n_max_features": 30, "feature_names": None, "excluded": [],
        "forced": [], "method": "classification", "groups": [],
        "abs_correlation": True, "n_min_features": 10, "verbose": False,
        "max_correlation": 0.4}, "estimator": {"verbose": False,
        "pairwise": True, "radius": None, "C": 1.0, "fit_intercept": True}},
        estimator=RobustLogisticRegression(C=1.0, fit_intercept=True,
        pairwise=True, radius=None, verbose=False),
        feature_selection=SelectOptimize(abs_correlation=True, excluded=[],
        feature_names=None, forced=[], groups=[], max_correlation=0.4,
        method="classification", n_max_features=30, n_min_features=10,
        verbose=False), feature_names=None, name="0001_0001", verbose=False)
        """
        if not self._is_run:
            raise NotRunException(self, "run")
        try:
            return next(m for m in self._models_classes if m.name == model_id)
        except Exception:
            raise ValueError("model_id: {} is not valid.".format(model_id))

    def results(self, step="run", format="dataframe"):
        """
        Return description and performance metrics for each model candidate
        generated.

        Parameters
        ----------
        step : str or None (default="run")
            Step name.

        format : str, "dataframe" or "json" (default="dataframe")
            If "dataframe" return pandas.DataFrame. Otherwise, return
            serialized json.

        Warning
        -------
        Timing measurements might *not* be accurate when running in
        parallel mode, ``n_jobs > 1``.
        """
        if step not in ("run"):
            raise ValueError("step not found.")

        if step == "run" and not self._is_run:
            raise NotRunException(self, "run")

        return _model_generator_results(self._models_metrics, format)
