"""
Sampling bias
"""

# Author: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2019.


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...core.base import GRMlabBase
from ...core.exceptions import NotRunException, NotFittedException
from ...data_analysis import Bivariate, Univariate
from ...modelling.base import GRMlabModel
from ...modelling.classification import GRMlabModelClassification
from ...modelling.model_optimization import ModelOptimizer


class stabilityIndex(GRMlabBase):
    """Calculates the PSI of two databases.

    It can use the binning from the bivariate class, and/or the categories or
    histogram form the univariate class.

    Parameters
    ----------
    name : str
        Name given to the class. This is only an identifier set by the
        user.

    kgb_bivariate : :class:`Bivariate`
        Bivariate fitted with KGB database. Used to replicate the bivariate
        transformation (binning, WoEs and remove variables) on the database
        part to be inferred.

    univariate : :class:`Univariate`
        Univariate run with the whole database.

    target : str (default=None)
        Name of the target column.

    variables : list (default=None)
        List of variables to obtain the PSI. If None, PSI for all variables is
        calculated or, if there is a model, only the PSI of the variables
        present in the model are used.

    kgb_model : :class:`GRMlabModel`, :class:`GRMlabModelClassification` or :class:`ModelOptimizer` (default=None)
        The KGB model used to calculate the initial scores for the inference.

    verbose : boolean (default=True)
        Controls verbosity of output.
    """
    def __init__(self, name, kgb_bivariate=None, univariate=None, target=None,
                 variables=None, kgb_model=None, verbose=True):
        # parameters
        self.name = name
        self.univariate = univariate
        self.kgb_bivariate = kgb_bivariate
        self.kgb_model = kgb_model
        self.target = target
        self.variables = variables
        self.verbose = verbose

        # run
        self._woe_df = None
        self.df_psi = None
        self.df_psi_univar = None
        # with model
        self._df_target = None
        self._df_score = None
        self._df_bivar_score = None
        self.score_psi = None
        self._conclusion = None

        # flag
        self._is_run = False

    def plot_var_psi(self, name):
        """Plots the PSI value of the univar categories or bins.
        """
        if not self._is_run:
            raise NotRunException(self, "run")
        if self.univariate is None:
            return None
        plt.title(name)
        plt.plot(self.df_psi_univar[name]["plot"]["cat"],
                 self.df_psi_univar[name]["plot"]["psi"])
        plt.ylabel("PSI per category")
        plt.xticks(rotation=90)
        plt.show()
        plt.close()

    def results(self):
        """Returns a dataframe with the results of the analysis.

        The columns are: name of the variable; PSI: obtained with the
        binning from the bivariate class; iv_ref: the value of the IV of the
        variable; Norm: values [0, 1, 2], 0 for no difference (PSI<0.1), 1 for 
        some difference (0.1<=PSI<0.3), and 2 for strong difference (0.3<=PSI);
        PSI univar: psi obtained with the distribution or the
        original categories from the univariate class; univar diff: category or
        distribution bin with the highest difference;
        PSI change: PSI univar / PSI - 1.
        """
        if not self._is_run:
            raise NotRunException(self, "run")
        return self.df_psi

    def run(self, df, df_ref, df_bivar):
        """
        Runs the inference model.

        It only returns the df if parameter self.reorder_df_records is set to
        True.

        Parameters
        ----------
        df : pandas.DataFrame, shape = [n_samples, n_features]
            The database from which target is going to be inferred. The
            preprocessing and univariate analysis have to be already executed
            on it.

        df_ref : pandas.DataFrame, shape = [n_samples*, n_features]
            The database used to model the KGB model. The preprocessing,
            univariate have to be already executed, but not the bivariate.

        df_bivar : pandas.DataFrame, shape = [n_samples_kgb, n_features_kgb]
            The database used to model the KGB model. The preprocessing,
            univariate and bivariate analysis must be already executed
            on it.
        """

        # copy df to avoid transforming the original df.
        self._woe_df = df.copy()

        # check univariate
        if self.univariate is not None:
            if not isinstance(self.univariate, (Univariate)):
                raise TypeError("class {} is not a univariate class.".
                                format(type(self.univariate).__name__))

        # check bivariate
        if self.kgb_bivariate is not None:
            if not isinstance(self.kgb_bivariate, (Bivariate)):
                raise TypeError("class {} is not a bivariate class.".
                                format(type(self.kgb_bivariate).__name__))

            # check model
            if self.kgb_model is not None:
                if isinstance(self.kgb_model, (GRMlabModelClassification,
                                               GRMlabModel)):
                    if not self.kgb_model._is_estimator_fitted:
                        raise NotFittedException(self.kgb_model)
                elif isinstance(self.kgb_model, ModelOptimizer):
                    self.kgb_model = self.kgb_model.best_model
                else:
                    raise TypeError("class {} is not an accepted model class.".
                                    format(type(self.kgb_model).__name__))
        else:
            # if no bivariate is given, the model is set to None.
            self.kgb_model = None

        # run all model inference
        self._run(df, df_ref, df_bivar)
        # flag
        self._is_run = True

    def _bivariate_psi(self, df_bivar, variables):
        """calculate the PSI for each variable with the binning given by
        the bivariate class
        """
        if self.kgb_bivariate is None:
            self.df_psi = pd.DataFrame(
                variables, columns=["name"]).set_index("name")
            return 0
        dict_psi = self._psi(df_bivar, variables)

        self.df_psi = pd.DataFrame.from_dict(
            {i: dict_psi[i] for i in dict_psi.keys()}, orient='index')
        df_iv = self.kgb_bivariate.results()[np.isin(
                    self.kgb_bivariate.results()["name"].values, variables)][
                    ["name", "iv"]].set_index("name")
        self.df_psi = pd.concat([self.df_psi, df_iv],
                                axis=1, sort=False
                                ).sort_values(
                                    by=['iv'], ascending=False
                                ).rename(columns={"iv": "iv_ref"})
        self.df_psi["Norm"] = np.digitize(
            self.df_psi["PSI"].values,
            np.array([0.1, 0.3]))

    def _check_and_homogenize_dfs(self, df_bivar):
        """check that df contains all columns in df_bivar and order them
        equally
        """
        if self.kgb_bivariate is None:
            return 0
        # delete extra columns in df
        to_delete = list(set(self._woe_df.columns).difference(
            set(df_bivar.columns)))
        self._woe_df = self._woe_df.drop(columns=to_delete)
        # check if all df_bivar columns are in df
        diff_cols = list(set(df_bivar.columns).difference(
            set(self._woe_df.columns)))
        if len(diff_cols) != 0:
            raise IndexError("columns {} are missing in dataframe to be"
                             " inferred".format(diff_cols))
        # order df columns in same order as df_bivar
        self._woe_df = self._woe_df[df_bivar.columns]

    def _decision(self):
        if self.score_psi is not None:
            if self.score_psi < 0.1:
                self._conclusion = (
                    "No difference detected. KGB model can be "
                    "directly applied to infer the target.")
            elif self.score_psi < 0.25:
                self._conclusion = (
                    "Difference detected. Inference may be necessary "
                    "to correctly infer the target.")
            elif self.score_psi > 0.25:
                self._conclusion = (
                    "Strong difference detected. Inference is highly "
                    "recommended to correctly infer the target.")
        else:
            pass

    def _kgb_bivariate_transform(self):
        """binning, woe and bivariate transformations to df"""
        if self.kgb_bivariate is None:
            return 0
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

    def _predict(self, df_bivar):
        if self.verbose:
            print("Predicting score for df with kgb....")
        # predict target with kgb model
        self._df_target = self.kgb_model.predict(self._woe_df.values)
        # predict probability with kgb model
        self._df_score = self.kgb_model.predict_proba(self._woe_df.values)
        # score based on prob of being good of df and kgb_df with kgb model
        self._df_bivar_score = self.kgb_model.predict_proba(df_bivar.values)
        if self.verbose:
            print("done!")

    def _psi(self, df_bivar, variables):
        dict_psi = {}
        for column in variables:
            count_ref = df_bivar[[column]].groupby(column).apply(
                lambda x: x.count()/df_bivar.shape[0]
                ).rename(columns={column: "count_ref"})
            count = self._woe_df[[column]].groupby(column).apply(
                lambda x: x.count()/self._woe_df.shape[0]
                ).rename(columns={column: "PSI"})
            df_count = pd.concat([count_ref, count],
                                 axis=1, join='inner')

            dict_psi[column] = {}
            dict_psi[column]["PSI"] = df_count.apply(
                lambda x: (x["count_ref"]-x["PSI"])*np.log(
                    x["count_ref"]/x["PSI"]), axis=1).values.sum()

        return dict_psi

    def _run(self, df, df_ref, df_bivar):
        # binning, woe and bivariate transformations to df
        self._kgb_bivariate_transform()
        # check that df contains all columns in df_kgb and order them equally
        self._check_and_homogenize_dfs(df_bivar)

        if self.kgb_model is not None:
            # score the df with the kgb_model
            self._predict(df_bivar)
            self._stability_score()
            if self.variables is None:
                model_vars = self.kgb_model.get_support(output_names=True)
                self._bivariate_psi(df_bivar, model_vars)
                # univariate PSI
                self._univariate_psi(df, df_ref, model_vars)
            else:
                self._bivariate_psi(df_bivar, self.variables)
                # univariate PSI
                self._univariate_psi(df, df_ref, self.variables)
        else:
            # calculate stability index
            if self.variables is not None:
                var_psi = self.variables
            else:
                var_psi = df_bivar.columns
            self._bivariate_psi(df_bivar, var_psi)
            # univariate PSI
            self._univariate_psi(df, df_ref, var_psi)

        # recommended action
        self._decision()

    def _stability_score(self):
        """calculate the PSI of the score given by the model.
        """
        hist_1 = np.histogram(
            self._df_score[:, 0],
            bins=np.arange(0, 1.1, 0.1))[0]/self._df_score.shape[0]
        hist_2 = np.histogram(
            self._df_bivar_score[:, 0],
            bins=np.arange(0, 1.1, 0.1))[0]/self._df_bivar_score.shape[0]
        self.score_psi = sum((hist_1-hist_2)*np.log(hist_1/hist_2))

    def _univariate_psi(self, df, df_ref, variables):
        """calculate the PSI for each variable with categories or histogram
        given by the univariate class.
        """
        if self.univariate is None:
            return None
        if self.verbose:
            print("Obtaining PSI of univariate categories....")
        self.df_psi_univar = {}
        for col in variables:
            if col in [self.univariate.target, self.univariate.date]:
                continue
            self.df_psi_univar[col] = {}
            # numerical variables
            if self.univariate._dict_variables[col]["dtype"] in [
                    "ordinal", "numerical"]:
                bins = self.univariate._dict_variables[col]["d_hist_pos"]
                # count elements in histogram
                df_bk = np.histogram(df[col].dropna(), bins=bins)
                df_ref_bk = np.histogram(df_ref[col].dropna(), bins=bins)
                # count outliers
                df_bk = np.append(
                    np.append([sum(df[col].dropna() < bins[0])], df_bk[0]),
                    [sum(df[col].dropna() > bins[-1])])
                df_ref_bk = np.append(
                    np.append([sum(df_ref[col].dropna() < bins[0])],
                              df_ref_bk[0]),
                    [sum(df_ref[col].dropna() > bins[-1])])
                # count nans
                df_bk = np.append(df_bk, len(df[col]) - df[col].count())
                df_ref_bk = (np.append(df_ref_bk, len(df_ref[col]) -
                             df_ref[col].count()))
                # avoid zeros
                df_bk += 1
                df_ref_bk += 1
                # normalize to one.
                df_bk = df_bk/np.sum(df_bk)
                df_ref_bk = df_ref_bk/np.sum(df_ref_bk)
                # calculate psi.
                ranges_psi = ([str(bins[i]) + "__" + str(bins[i+1])
                               for i in range(len(bins)-1)] +
                              ["outliers_low", "outliers_high", "missings"])
                vect_psi = [(elt - df_ref_bk[i]) * np.log(elt / df_ref_bk[i])
                            for i, elt in enumerate(df_bk)]
                self.df_psi_univar[col]["psi"] = sum(vect_psi)
                self.df_psi_univar[col]["most_diff"] = ranges_psi[
                    np.argmax(vect_psi)]
                self.df_psi_univar[col]["plot"] = {"cat": ranges_psi,
                                                   "psi": vect_psi}

            # categorical variables
            elif self.univariate._dict_variables[col]["dtype"] in [
                    "categorical", "nominal"]:
                    # count categories
                df_bk = np.unique(df[col].dropna(), return_counts=True)
                df_ref_bk = np.unique(df_ref[col].dropna(), return_counts=True)
                # common categories
                all_cats = set(df_bk[0]).union(set(df_ref_bk[0]))
                dict_all_cats = dict(zip(all_cats, [1]*len(all_cats)))
                # create dict
                df_dict = dict_all_cats.copy()
                df_ref_dict = dict_all_cats.copy()
                for i, elt in enumerate(df_bk[0]):
                    df_dict[elt] += df_bk[1][i]
                for i, elt in enumerate(df_ref_bk[0]):
                    df_ref_dict[elt] += df_ref_bk[1][i]
                # count nans
                df_dict["missings"] = (len(df[col]) - df[col].count()) + 1
                df_ref_dict["missings"] = (len(df_ref[col]) -
                                           df_ref[col].count()) + 1
                # change to percentage
                df_tot = sum(df_dict.values())
                df_ref_tot = sum(df_ref_dict.values())
                df_dict = {k: df_dict[k] / df_tot for k in df_dict}
                df_ref_dict = {k: df_ref_dict[k] / df_ref_tot
                               for k in df_ref_dict}
                # PSI
                vect_psi = [(df_dict[key] - df_ref_dict[key]) *
                            np.log(df_dict[key] / df_ref_dict[key])
                            for key in df_ref_dict]

                self.df_psi_univar[col]["psi"] = sum(vect_psi)
                self.df_psi_univar[col]["most_diff"] = list(
                    df_ref_dict.keys())[np.argmax(vect_psi)]
                self.df_psi_univar[col]["plot"] = {"cat": df_ref_dict.keys(),
                                                   "psi": vect_psi}

        # add univariate info
        univar_psi = []
        univar_diff = []
        for elt in self.df_psi.index:
            if elt in [self.univariate.target, self.univariate.date]:
                univar_psi.append(None)
                univar_diff.append(None)
            else:
                univar_psi.append(self.df_psi_univar[elt]["psi"])
                univar_diff.append(self.df_psi_univar[elt]["most_diff"])

        self.df_psi["PSI univar"] = univar_psi
        self.df_psi["univar diff"] = univar_diff
        if self.kgb_bivariate is not None:
            self.df_psi["PSI change"] = (self.df_psi["PSI univar"] /
                                         self.df_psi["PSI"] - 1)

        if self.verbose:
            print("done!")
