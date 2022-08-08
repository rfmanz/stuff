"""
Inference models
"""

# Author: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2019.


import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.utils import check_X_y

from ...core.base import GRMlabBase
from ...core.exceptions import NotRunException, NotFittedException
from ...data_analysis import Bivariate
from ...modelling.base import GRMlabModel
from ...modelling.classification import GRMlabModelClassification
from ...modelling.model_optimization import ModelOptimizer


class Inferencer(GRMlabBase):
    """Infers the target of a given database.

    Two  models are available for inference: the parceling model and
    the fuzzy model.

    Parameters
    ----------
    name : str
        Name given to the inferencer. This is only an identifier set by the
        user.

    inference_model : str
        Two options available, parceling or fuzzy.

    kgb_model : :class:`GRMlabModel`, :class:`GRMlabModelClassification` or :class:`ModelOptimizer`
        The KGB model used to calculate the initial scores for the inference.

    kgb_bivariate : :class:`Bivariate`
        Bivariate fitted with KGB database. Used to replicate the bivariate
        transformation (binning, WoEs and remove variables) on the database
        part to be inferred.

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

    reorder_df_records : boolean (default=True)
        If True, the entries of the input database will be ordered or/and
        expanded according to the model used. If False, no modifications
        will be implemented on the input database. For parceling, the
        reordering of records is compulsory.

    add_target : boolean (default=True)
        If True, the inferred target will be added to the input database. The
        parameter reorder_df_records must also be true.

    train_model : boolean (default=True)
        If True, the same type of model as the kgb_model will be trained with
        the complete database obtained with the inference.

    fit_feature_selection : boolean (default=True)
        Only needed when train_model=True. If False, the feature selection
        variables in the model with the complete database will be the same as
        the ones used in the KGB model. If True, the feature selection will
        be executed again and different variables might appear in the model.

    random_state : int or None (default=None)
        The seed used by the random number generator.

    verbose : boolean (default=True)
        Controls verbosity of output.
    """
    def __init__(self, name, inference_model, kgb_model, kgb_bivariate,
                 kgb_class_weights, reject_rate, ERI=1, num_buckets=10,
                 target=None, reorder_df_records=True, add_target=True,
                 train_model=False, fit_feature_selection=True,
                 random_state=None, verbose=True):

        # general parameters
        self.name = name
        self.target = target
        # parameters of the kgb database
        self.kgb_model = kgb_model
        self.kgb_bivariate = kgb_bivariate
        self.kgb_class_weights = kgb_class_weights
        # parameters for the inference model
        self.inference_model = inference_model
        self.reject_rate = reject_rate
        self.eri = ERI
        self.num_buckets = num_buckets
        # Boolean parameters
        self.verbose = verbose
        self.reorder_df_records = reorder_df_records
        self.add_target = add_target
        self.fit_feature_selection = fit_feature_selection
        self.train_model = train_model

        # variables obtained directly applying the kgb model on the inferred df
        # without using any inferring model.
        self._kgb_target = None
        self._kgb_class_prob = None

        # variables of the inferred database
        self._weight_df = None
        self._woe_df = None
        self._inferred_target = []

        # parceling variables
        self.bucket_bad_rate = []
        self.bucket_cuts = []

        # kgb model score histogram
        self.pd_rejected = None
        self.hist_pd_rejected = None
        self.pd_kgb = None
        self.hist_pd_kgb = None
        # inference model socore histogram
        self.pd_inf_rejected = None
        self.hist_inf_pd_rejected = None
        self.pd_inf_kgb = None
        self.hist_inf_pd_kgb = None

        # new model fitted on the complete database
        self.model = None

        # flags
        self._is_run = False

        if random_state is not None:
            random.seed(random_state)

    def get_target(self):
        """Return the inferred target of the input database

        Be aware that the order of the target entries may not be the same
        as the entries order of the original input df.
        """
        if not self._is_run:
            raise NotRunException(self)
        return self._inferred_target

    def run(self, df, df_kgb, target_kgb):
        """
        Runs the inference model.

        It only returns the df if parameter self.reorder_df_records is set to
        True.

        Parameters
        ----------
        df : pandas.DataFrame, shape = [n_samples, n_features]
            The database which target is going to be inferred. The
            preprocessing and univariate analysis have to be already executed
            on it.

        df_kgb : pandas.DataFrame, shape = [n_samples_kgb, n_features_kgb]
            The database used to model the KGB model. The preprocessing,
            univariate and bivariate analysis have to be already executed
            on it.

        target_kgb : array-like, shape = [n_samples_kgb]
            Target of the df_kgb database.
        """
        # copy df to avoid transforming the original df.
        self._woe_df = df.copy()

        # check model
        if isinstance(self.kgb_model, (GRMlabModelClassification,
                                       GRMlabModel)):
            if not self.kgb_model._is_estimator_fitted:
                raise NotFittedException(self.kgb_model)
        elif isinstance(self.kgb_model, ModelOptimizer):
            self.kgb_model = self.kgb_model.best_model
        else:
            raise TypeError("class {} is not an accepted model class.".
                            format(type(self.kgb_model).__name__))

        # check bivariate
        if not isinstance(self.kgb_bivariate, (Bivariate)):
            raise TypeError("class {} is not a bivariate class.".
                            format(type(self.kgb_bivariate).__name__))

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

        # check shape of df_kgb and target_kgb
        check_X_y(df_kgb, target_kgb)

        # run all model inference
        df = self._run(df, df_kgb, target_kgb)
        # flag
        self._is_run = True

        # only return the df if a reordering of its entries has been
        # implemented.
        if self.reorder_df_records:
            return df, self._weight_df

    def stats(self):
        """Plots the figures shown with verbose=True.
        """
        if not self._is_run:
            raise NotRunException(self)

        if self.inference_model == "parceling":

            norm_bins = [(elt+self.bucket_cuts[i])/2 for i, elt in enumerate(
                self.bucket_cuts[1:])]
            plt.bar(norm_bins, self.bucket_bad_rate, width=0.05,
                    align="center", alpha=0.6, label="parceling")
            plt.bar(norm_bins, [elt/self.eri for elt in self.bucket_bad_rate],
                    width=0.05, align="center", alpha=0.6, label="kgb")
            plt.title("bucket default rate")
            plt.legend()
            plt.show()
            plt.close()

        self._plot_hist_score(
            [
                [self.pd_rejected, self.hist_pd_rejected],
                [self.pd_kgb, self.hist_pd_kgb]
            ],
            ["df", "df_kgb"], "score distribution based on the kgb model")

        if self.train_model:
            self._plot_hist_score(
                [
                    [self.pd_inf_rejected, self.hist_inf_pd_rejected],
                    [self.pd_inf_kgb, self.hist_inf_pd_kgb]
                ],
                ["df", "df_kgb"], "score distribution based on"
                " inferenced model")

    def _run(self, df, df_kgb, target_kgb):
        # binning, woe and bivariate transformations to df
        self._kgb_bivariate_transform()
        # check that df contains all columns in df_kgb and order them equally
        self._check_and_homogenize_dfs(df_kgb)
        # score the df with the kgb_model
        self._predict()
        # plot distribution of df and df_kgb to compare them
        bins = np.arange(0, 1.1, 0.1)
        self.pd_rejected, self.hist_pd_rejected = np.histogram(
            self._kgb_class_prob[:, 0], bins=bins)
        self.pd_kgb, self.hist_pd_kgb = np.histogram(
            self.kgb_model.predict_proba(df_kgb.values)[:, 0],
            bins=bins)
        if self.verbose:
            self._plot_hist_score(
                [
                    [self.pd_rejected, self.hist_pd_rejected],
                    [self.pd_kgb, self.hist_pd_kgb]
                ],
                ["df", "df_kgb"], "score distribution based on the kgb model")
        # inference model based on the score given by the kgb_model
        df = self._inference_model(df, df_kgb, target_kgb)
        if self.train_model:
            # New model with the complete database. The feature selection and
            # model estimator have the same parameters as the kgb_model.
            self._fit_model(df_kgb, target_kgb)
            # plot distribution of df and df_kgb to compare them with the score
            # given with the new model.
            bins = np.arange(0, 1.1, 0.1)
            self.pd_inf_rejected, self.hist_inf_pd_rejected = np.histogram(
                self.model.predict_proba(
                    self._woe_df[df_kgb.columns.values].values)[:, 0],
                bins=bins)
            self.pd_inf_kgb, self.hist_inf_pd_kgb = np.histogram(
                self.model.predict_proba(df_kgb.values)[:, 0],
                bins=bins)
            if self.verbose:
                self._plot_hist_score(
                    [
                        [self.pd_inf_rejected, self.hist_inf_pd_rejected],
                        [self.pd_inf_kgb, self.hist_inf_pd_kgb]
                    ],
                    ["df", "df_kgb"], "score distribution based on"
                    " inferenced model")
        return df

    def _check_and_homogenize_dfs(self, df_kgb):
        """check that df contains all columns in df_kgb and order them equally
        """
        # delete extra columns in df
        to_delete = list(set(self._woe_df.columns).difference(
            set(df_kgb.columns)))
        self._woe_df = self._woe_df.drop(columns=to_delete)
        # check if all df_kgb columns are in df
        diff_cols = list(set(df_kgb.columns).difference(
            set(self._woe_df.columns)))
        if len(diff_cols) != 0:
            raise IndexError("columns {} are missing in dataframe to be"
                             " inferred".format(diff_cols))
        # order df columns in same order as df_kgb
        self._woe_df = self._woe_df[df_kgb.columns]

    def _fuzzy_method(self, df, df_kgb, target_kgb):
        """Fuzzy method

        The event rate increase (ERI) is the only degree of freedom.
        """
        # kgb weight
        kgb_weight = 0
        for key in self.kgb_class_weights:
            kgb_weight += (len(target_kgb[target_kgb == key]) *
                           self.kgb_class_weights[key])
        # inferred weight
        inferred_weight = ((self.reject_rate/(1-self.reject_rate)) /
                           (len(self._kgb_target)/kgb_weight))
        # fuzzy weight
        inferred_fuzzy_weight = self._kgb_class_prob * inferred_weight
        inferred_fuzzy_weight[:, 1] = inferred_fuzzy_weight[:, 1] * self.eri
        # generate X, y and weight for df.
        # weights for df
        self._weight_df = inferred_fuzzy_weight.flatten('F')
        if self.verbose:
            print("   Weights for df obtained!")
        # inferred target for df
        self._inferred_target = (np.array([0]*self._woe_df.shape[0] +
                                 [1]*self._woe_df.shape[0]))
        if self.verbose:
            print("   Target for df inferred!")
        # array of df
        if self.train_model:
            self._woe_df = self._woe_df.append(
                self._woe_df, ignore_index=True, verify_integrity=True)
        if self.reorder_df_records:
            df = df.append(df, ignore_index=True, verify_integrity=True)
            if self.add_target:
                df[self.target] = self._inferred_target
        return df

    def _inference_model(self, df, df_kgb, target_kgb):
        """inference model based on the score given by the kgb_model"""
        if self.verbose:
            print("Starting {} inference model....".format(
                self.inference_model))

        if self.inference_model == "fuzzy":
            df = self._fuzzy_method(df, df_kgb, target_kgb)

        elif self.inference_model == "parceling":
            df = self._parceling_method(df, df_kgb, target_kgb)

        return df

    def _kgb_bivariate_transform(self):
        """binning, woe and bivariate transformations to df"""
        if self.verbose:
            print("Transforming df with bivariate class....")
            print("   WoE transformation....")
        # WoEs transformation
        self.kgb_bivariate.optimalgrouping.transform(
            self._woe_df, only_metric=True, add_target=False)
        if self.verbose:
            print("   WoE transformation done!")
            print("   deleting variables....")
        # Delete varaibles
        self.kgb_bivariate.transform(data=self._woe_df, mode="basic")
        if self.verbose:
            print("   done!")

    def _parceling_method(self, df, df_kgb, target_kgb):
        """Parceling method

        The event rate increase (ERI), and the number of buckets are the two
        degrees of freedom.
        """
        # kgb weight
        kgb_weight = 0
        for key in self.kgb_class_weights:
            kgb_weight += (len(target_kgb[target_kgb == key]) *
                           self.kgb_class_weights[key])
        # inferred weight
        inferred_weight = ((self.reject_rate/(1-self.reject_rate)) /
                           (len(self._kgb_target)/kgb_weight))
        # weights for all entries
        self._weight_df = [inferred_weight]*self._woe_df.shape[0]
        if self.verbose:
            print("   Weights for df obtained!")

        # score based on probability of being good of df and kgb_df with kgb
        # model
        score_kgb = self.kgb_model.predict_proba(df_kgb.values)[:, 0]
        score = self._kgb_class_prob[:, 0]
        # generate buckets of scores (optimal grouping might be used)
        bucket_increment = (max(score_kgb)-min(score_kgb))/self.num_buckets
        self.bucket_cuts = [0.] + [
            (min(score_kgb) + bucket_increment*(i+1))
            for i in range(self.num_buckets-1)] + [1.]
        # calculate bad rate for each bucket
        self.bucket_bad_rate = [
            min(1., (target_kgb[(
                score_kgb >= self.bucket_cuts[i]) & (
                    score_kgb < self.bucket_cuts[i+1])
                ].mean())*self.eri) for i in range(self.num_buckets)]
        if self.verbose:
            print("   Backet bad rate obtained!")
            norm_bins = [(elt+self.bucket_cuts[i])/2 for i, elt in enumerate(
                         self.bucket_cuts[1:])]
            plt.bar(norm_bins, self.bucket_bad_rate, width=0.05,
                    align="center", alpha=0.6, label="parceling")
            plt.bar(norm_bins, [elt/self.eri for elt in self.bucket_bad_rate],
                    width=0.05, align="center", alpha=0.6, label="kgb")
            plt.title("bucket default rate")
            plt.legend()
            plt.show()
            plt.close()
        # randomly infer the target for df entries depending on bucket
        if self.train_model:
            woe_df_copia = pd.DataFrame()
        df_reorder = pd.DataFrame()
        for i in range(self.num_buckets):
            buck_attibutes = self._woe_df[(score >= self.bucket_cuts[i]) &
                                          (score < self.bucket_cuts[i+1])
                                          ][df_kgb.columns]
            if self.train_model:
                woe_df_copia = woe_df_copia.append(
                    buck_attibutes, ignore_index=True, verify_integrity=True)
            df_reorder = df_reorder.append(
                df[(score >= self.bucket_cuts[i]) &
                   (score < self.bucket_cuts[i+1])], ignore_index=True)
            self._inferred_target.append(
                (np.array([random.random() for _ in range(
                    buck_attibutes.shape[0])]) <= self.bucket_bad_rate[i]
                 ).astype(int))
        if self.train_model:
            self._woe_df = woe_df_copia
        self._inferred_target = np.concatenate(self._inferred_target)
        if self.verbose:
            print("   Target for df inferred!")
        if self.add_target:
            df_reorder[self.target] = self._inferred_target
        return df_reorder

    def _predict(self):
        if self.verbose:
            print("Predicting score for df with kgb....")
        # predict target with kgb model
        self._kgb_target = self.kgb_model.predict(self._woe_df.values)
        # predict probability with kgb model
        self._kgb_class_prob = self.kgb_model.predict_proba(
            self._woe_df.values)
        if self.verbose:
            print("done!")

    def _plot_hist_score(self, score_vec, names, title):
        for i, [vec, bins_vec] in enumerate(score_vec):
            norm_vec = vec / vec.sum()
            norm_bins = [(elt+bins_vec[i])/2 for i, elt in enumerate(
                bins_vec[1:])]
            plt.bar(norm_bins, norm_vec, width=0.05, align="center",
                    alpha=0.6, label=names[i])
        plt.title(title)
        plt.legend()
        plt.show()
        plt.close()

    def _generate_complete_data(self, df_kgb, target_kgb):
        """Generates X, y and weight of complete db.

        Parameters
        ----------
        df_kgb : pandas.DataFrame, shape = [n_samples_kgb, n_features_kgb]
            Database used to model the KGB model, however, any other database
            with the same columns names is valid. The preprocessing,
            univariate and bivariate analysis have to be already executed
            on it. The target should not be present, as is contained in
            parameter `target_kgb`.

        target_kgb : array-like, shape = [n_samples_kgb]
            Target of the df_kgb database.
        """
        # weights for all entries
        weight = np.concatenate([np.full(len(target_kgb), 1.),
                                 self._weight_df])
        # target for all entries
        y = np.concatenate([target_kgb, self._inferred_target])
        # complete dataframe
        X = df_kgb.append(
                self._woe_df, ignore_index=True,
                verify_integrity=True)[df_kgb.columns].values
        return X, y, weight

    def _fit_model(self, df_kgb, target_kgb):
        """New model with the complete database.

        The feature selection and model estimator have the same parameters
        as the kgb_model.

        Parameters
        ----------
        df_kgb : pandas.DataFrame, shape = [n_samples_kgb, n_features_kgb]
            Database used to model the KGB model, however, any other database
            with the same columns names is valid. The preprocessing,
            univariate and bivariate analysis have to be already executed
            on it. The target should not be present, as is contained in
            parameter `target_kgb`.

        target_kgb : array-like, shape = [n_samples_kgb]
            Target of the df_kgb database.
        """
        X, y, weight = self._generate_complete_data(df_kgb, target_kgb)

        if self.verbose:
            print("   Obtaining new model....")
        # initialize new model with same characteristics as the kgb model.
        self.model = GRMlabModelClassification(
            name="auto_" + self.inference_model,
            feature_selection=copy.deepcopy(self.kgb_model.feature_selection),
            estimator=copy.deepcopy(self.kgb_model.estimator)
            )

        if (self.fit_feature_selection and
                (self.model.feature_selection is not None)):
            # fit the feature selection and estimator
            self.model.feature_selection.feature_names = df_kgb.columns.values
            self.model.feature_selection = self.model.feature_selection.fit(
                X, y)
            if self.verbose:
                print("      New feature selection fitted!")
            self.model._is_feature_selection_fitted = True
        if self.model.feature_selection is None:
            self.model.estimator = self.model.estimator.fit(
                X=X, y=y,
                sample_weight=weight)
        else:
            self.model.estimator = self.model.estimator.fit(
                X=self.model.feature_selection.transform(X), y=y,
                sample_weight=weight)
            self.model._is_feature_selection_fitted = True
        if self.verbose:
                print("      Estimator fitted!")

        self.model._is_estimator_fitted = True
        if self.verbose:
            print("done!")
