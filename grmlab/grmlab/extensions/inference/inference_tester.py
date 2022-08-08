"""
Inference Tester
"""

# Author: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2019.


import random

from sklearn.utils import check_X_y

from .inferencer import Inferencer
from ...core.base import GRMlabBase
from ...core.exceptions import NotRunException, NotFittedException
from ...data_analysis import Bivariate
from ...data_processing.preprocessing import Preprocessing
from ...modelling.base import GRMlabModel
from ...modelling.classification import GRMlabModelClassification
from ...modelling.model_optimization import ModelOptimizer
from ...modelling.model_analysis import ModelAnalyzer


class Tester(GRMlabBase):
    """Test the performance of the inference models.

    1- The old_model has been trained with old data, and is used to accept or
    reject more recent data. The accept, reject status is passed with
    flag_reject in the :mod:`run` method.
    2- A new model (self.accepted_model) is trained with these accepted
    records.
    3- self.accepted_model is use to infer the rejected records.
    4- The last model (self.infer_model) is trained with the accepted
    applicants and the inferred rejected applicants.
    5- Model analysis of these two models are executed to compare them.

    Parameters
    ----------
    name : str
        Name given to the inferencer. This is only an identifier set by the
        user.

    inference_model : str
        Two options available, parceling or fuzzy.

    old_model : :class:`GRMlabModel`, :class:`GRMlabModelClassification` or :class:`ModelOptimizer`
        The model used to split the applicants into rejected or accepted.

    old_bivariate : :class:`Bivariate`
        Bivariate used in the old model database. Used as template for the new
        bivariate classes initialization.

    kgb_class_weights : dict
        The weight given to each target element, for instance::

            kgb_class_weights = {0: 1.0, 1: 1.0}

        meaning that the same weights are given to the good and bad when
        fitting the model.

    reject_rate : float
        Expected reject rate in the database. The number must be in the range
        (0,1).

    ERI : int or float (default=1)
        Event rate increase. Expected increase in defaults on the database to
        be inferred relative to the KGB entries. For instance, ERI=1.5 means a
        50% increase, and ERI=1 means no increase.

    num_buckets : int (default=10)
        Number of buckets for the parceling model. Variable not needed in the
        other inference models.

    target : str
        Name of the target column.

    date : str or None (default=None)
        Name of the variable flagged as date.

    feature_selection : object (default=None)
        Feature selection algorithm with methods :mod:`fit`,
        :mod:`fit_transform` and :mod:`transform`, and attribute
        :mod:`support_`.

    estimator : object (default=None)
        A supervised learning estimator with a :mod:`fit` method.

    random_state : int or None (default=None)
        Seed used by the random number generator.

    verbose : boolean (default=True)
        It controls verbosity of output.
    """
    def __init__(self, name, inference_model, old_model, old_bivariate,
                 kgb_class_weights, reject_rate, ERI=1, num_buckets=10,
                 target=None, date=None, feature_selection=None,
                 estimator=None, random_state=None, verbose=True):

        # general parameters
        self.name = name
        self.target = target
        self.date = date
        self.random_state = random_state
        # parameters of the kgb database
        self.old_model = old_model
        self.old_bivariate = old_bivariate
        self.kgb_class_weights = kgb_class_weights
        # parameters for the inference model
        self.inference_model = inference_model
        self.reject_rate = reject_rate
        self.eri = ERI
        self.num_buckets = num_buckets
        # parameters for the model
        self.feature_selection = feature_selection
        self.estimator = estimator
        # boolean parameters
        self.verbose = verbose
        self.reorder_df_records = True
        self.add_target = False
        self.fit_feature_selection = False
        self.train_model = False

        if self.random_state is not None:
            random.seed(self.random_state)

        # run
        self.flag_reject = None
        self.accepted_model = None
        self.accepted_bivar = None
        self.inference = None
        self.infer_bivar = None
        self.infer_model = None
        self.analysis_accepted = None
        self.analysis_inference = None

        # flags
        self._is_run = False

    def run(self, df_kgb, target_kgb, flag_reject, dates_vals=None):
        """
        Run the inference model.

        It only returns the df if parameter self.reorder_df_records is set to
        True.

        Parameters
        ----------
        df_kgb : pandas.DataFrame, shape = [n_samples_kgb, n_features_kgb]
            The database used to model the KGB model. The preprocessing and
            univariate must be already executed on it, but not the
            bivariate.

        target_kgb : array-like, shape = [n_samples_kgb]
            Target of the df_kgb database.

        flag_reject : array-like, shape = [n_samples_kgb]
            Reject flag for all records of df_kgb database.

        dates_vals : array-like, shape = [n_samples_kgb], (default=None)
            Dates for all records of df_kgb database.
        """

        # check model
        if isinstance(self.old_model,
                      (GRMlabModelClassification, GRMlabModel)):
            if not self.old_model._is_estimator_fitted:
                raise NotFittedException(self.old_model)
        elif isinstance(self.old_model, ModelOptimizer):
            self.old_model = self.old_model.best_model
        else:
            raise TypeError("class {} is not an accepted model class.".
                            format(type(self.old_model).__name__))

        # check bivariate
        if not isinstance(self.old_bivariate, (Bivariate)):
            raise TypeError("class {} is not a bivariate class.".
                            format(type(self.old_bivariate).__name__))

        # check class-weights
        if not isinstance(self.kgb_class_weights, dict):
            raise TypeError("kgb_class_weights must be dict not {}".
                            format(type(self.kgb_class_weights).__name__))
        elif not (set(self.kgb_class_weights.keys()) == set(target_kgb)):
            raise ValueError(
                "the keys {} in class weights do not correspond to"
                " the target values".format(
                    set(self.kgb_class_weights.keys())))
        elif not(all(isinstance(elt, (float, int)) for elt in list(
                 self.kgb_class_weights.values()))):
            raise ValueError("values of kgb_class_weight must be numbers")

        # check inference model
        if self.inference_model not in ["parceling", "fuzzy"]:
            raise ValueError("Inference model {} no supported".format(
                self.inference_model))

        # check reject rate
        if not (self.reject_rate > 0 and self.reject_rate < 1):
            raise ValueError("reject rate must be in range (0,1)")

        # check ERI
        if self.eri <= 0:
            raise ValueError("ERI must be greater than 0")

        # check num_buckets
        if not isinstance(self.num_buckets, int):
            raise TypeError("num_buckets must be integer type")
        elif self.num_buckets <= 0:
            raise ValueError("num_buckets must be greater than 0")

        # check parceling and reorder_df_records
        if not self.reorder_df_records and self.inference_model == "parceling":
            raise ValueError("reorder_df_records must be true for parceling"
                             " model.")

        # check shape of df_kgb, target_kgb, flag_reject and dates_vals
        check_X_y(flag_reject.reshape(-1, 1), target_kgb)
        if dates_vals is not None:
            check_X_y(flag_reject.reshape(-1, 1), dates_vals)
        if df_kgb.shape[0] != len(flag_reject):
            ValueError("Found input variables with inconsistent numbers of "
                       "samples: [{},{}]".format(df_kgb.shape[0],
                                                 len(flag_reject)))

        # run all model inference
        self._run(df_kgb, target_kgb, dates_vals, flag_reject)
        # flag
        self._is_run = True

    def _run(self, df_kgb, target_kgb, dates_vals, flag_reject):
        print("split rejects and accepts")
        df_accepted, df_rejected = self._split_reject_accepted(
            df_kgb, flag_reject)

        print("obtain bivariate and model of accepts")
        self.accepted_bivar, self.accepted_model = self._bivar_model(
            df_accepted, "accepted_model")

        # Model analysis
        print("model analysis")
        df_kgb_copia = df_kgb.copy()
        self.accepted_bivar.optimalgrouping.transform(
            df_kgb_copia, only_metric=True, add_target=False)
        self.accepted_bivar.transform(data=df_kgb_copia, mode="basic")
        df_kgb_copia = df_kgb_copia[
                self.accepted_model.feature_selection.feature_names]
        X = df_kgb_copia.values
        y = df_kgb_copia[self.target].values
        self.analysis_accepted = ModelAnalyzer(
            name="test_name", model=self.accepted_model, n_simulations=100,
            simulation_size=0.3, feature_analysis_approximation=False,
            verbose=True)
        self.analysis_accepted.run(X=X, y=y, dates=dates_vals)

        # Reject Inference
        print("inference of rejected")
        self.inference = Inferencer(
            name="inference", inference_model=self.inference_model,
            kgb_model=self.accepted_model, kgb_bivariate=self.accepted_bivar,
            kgb_class_weights=self.kgb_class_weights,
            reject_rate=self.reject_rate, ERI=self.eri,
            num_buckets=self.num_buckets, verbose=self.verbose,
            target=self.target,
            train_model=self.train_model,
            fit_feature_selection=self.fit_feature_selection,
            random_state=self.random_state,
            reorder_df_records=self.reorder_df_records)

        df_infer, record_weight = self.inference.run(
            df=df_rejected, df_kgb=df_accepted[df_accepted.columns[:-1]],
            target_kgb=df_accepted[df_accepted.columns[-1]].values)

        df_all = df_kgb[self.flag_reject == 0].append(
            df_infer, ignore_index=True, verify_integrity=True, sort=True)

        print("obtain bivariate and model of accepts and rejects")
        self.infer_bivar, self.infer_model = self._bivar_model(
            df_all, "inference_model")

        # Model analysis Inference
        print("model analysis")
        df_kgb_copia = df_kgb.copy()
        self.infer_bivar.optimalgrouping.transform(
            df_kgb_copia, only_metric=True, add_target=False)
        self.infer_bivar.transform(data=df_kgb_copia, mode="basic")
        df_kgb_copia = df_kgb_copia[
                self.infer_model.feature_selection.feature_names]
        X = df_kgb_copia.values
        y = df_kgb_copia[self.target].values
        self.analysis_inference = ModelAnalyzer(
            name="test_name", model=self.infer_model, n_simulations=100,
            simulation_size=0.3, feature_analysis_approximation=False,
            verbose=True)
        self.analysis_inference.run(X=X, y=y, dates=dates_vals)

    def _split_reject_accepted(self, df_kgb, flag_reject):
        self.flag_reject = flag_reject
        df_accepted = df_kgb[self.flag_reject == 0]
        df_rejected = df_kgb[self.flag_reject == 1]

        return df_accepted, df_rejected

    def _bivar_model(self, df, name):
        # Preprocessing to eliminate potential constants
        print("preprocessing")
        prepro = Preprocessing(
            target=self.old_bivariate.target, date=self.old_bivariate.date,
            special_values=self.old_bivariate.special_values,
            verbose=False)
        prepro.run(df)
        prepro.transform(data=df, mode="basic")
        # Bivariate
        print("bivariate")
        bivar = Bivariate(
            target=self.old_bivariate.target, date=self.old_bivariate.date,
            special_values=self.old_bivariate.special_values, verbose=False,
            variables_nominal=self.old_bivariate.variables_nominal,
            max_gini=self.old_bivariate.max_gini,
            min_gini=self.old_bivariate.min_gini,
            max_iv=self.old_bivariate.max_iv, min_iv=self.old_bivariate.min_iv,
            max_pvalue_chi2=self.old_bivariate.max_pvalue_chi2,
            min_cramer_v=self.old_bivariate.min_cramer_v,
            monotonicity_force=self.old_bivariate.monotonicity_force)
        # run and transform bivariate
        bivar.run(data=df)
        bivar.optimalgrouping.transform(
            df, only_metric=True, add_target=True)
        bivar.transform(data=df, mode="basic")
        bivar.stats(step="transform")

        vars_names = df[df.columns[:-1]].columns.values
        X = df[df.columns[:-1]].values
        y = df[df.columns[-1]].values

        # Model
        print("model")
        self.feature_selection.feature_names = vars_names
        model = self.old_model
        model.feature_selection = self.feature_selection
        model.estimator = self.estimator
        model.name = name

        model.fit(X, y)

        return bivar, model
