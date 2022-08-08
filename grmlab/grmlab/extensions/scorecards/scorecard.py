"""
Credit risk scorecard.
"""

# Author: Carlos González Berrendero <c.gonzalez.berrender@bbva.com>
#         Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import copy
import os
import pickle
import platform

import numpy as np
import pandas as pd

from ...core.base import GRMlabBase
from ...core.exceptions import NotFittedException
from ...data_analysis.bivariate import Bivariate
from ...data_processing.feature_binning import OptimalGrouping
from ...modelling.classification import GRMlabModelClassification


class Scorecard(GRMlabBase):
    """
    Class to generate scorecards and export them in the standard
    format.

    Parameters
    ----------
    model : object (default=None)
        A GRMlabModelClassification class. Otherwise, an exception
        will be raised during fit.

    grouping : object
        A Bivariate or OptimalGrouping class. Otherwise, an exception
        will be raised during fit.

    verbose : int or bool (default=False)
        Controls verbosity of output.

    """
    def __init__(self, model=None, grouping=None, verbose=False):
        self.model = model
        self.grouping = grouping
        self.verbose = verbose

        # auxiliary
        self._optimalgrouping = None

        # flag
        self._is_fitted = False

    def fit(self):
        """
        Build scorecard with the model and grouping of the scorecard class.

        Returns
        -------
        self : object
        """
        return self._fit()

    def predict(self, df, from_woe=False):
        """
        Score a DB based on the scorecard.
        Scorecard must be fitted.

        Parameters
        ----------
        df : pandas.DataFrame
            It contains (at least) the features used by the model.

        from_woe : bool (default=False)
            True if df contains WoEs of the features. False otherwise.

        Returns
        -------
        final_score : numpy.ndarray,array of shape [n_samples]
            The predicted score values.

        """
        if not self._is_fitted:
            raise NotFittedException(self)

        final_score = np.zeros(df.shape[0])

        feature_names = self.model.get_support(output_names=True)

        for feature in feature_names:
            params = self._table[self._table["Name"] == feature][[
                "WoE", "Score"]].drop_duplicates().values

            woes, scores = params[:, 0], params[:, 1]

            # extract dataframe data
            data = df[feature].values

            # if data is not transformed to WoE, apply transformation before
            # computing final score.
            if not from_woe:
                data = self._transform(data, feature)

            for i, woe in enumerate(woes):
                final_score[abs(data - woe) < 1e-5] += scores[i]

        return final_score

    def table(self):
        """
        Return scorecard table.

        Returns
        -------
        table : pandas.DataFrame
           The scorecard in the standard format.
        """
        if not self._is_fitted:
            raise NotFittedException(self)

        return self._table

    def _to_excel(self, excel_writer, sheet_name="Scorecard"):
        """
        Export scorecard to excel.
        We recommend use the function export to get the scorecard. This is
        the reason for definning this function as private.

        Parameters
        ----------
        excel_writer : str or ExcelWriter object
            File path or existing ``pandas.ExcelWriter``. The scorecard will be
            exported to excel file.

        sheet_name : str (default="Scorecard")
            Name of the sheet which will contain the scorecard.
        """
        if not isinstance(excel_writer, (pd.ExcelWriter, str)):
            raise TypeError("excel_writer must be a string or ExcelWriter.")

        if not isinstance(sheet_name, str):
            raise TypeError("sheet_name must be a string.")

        self._table.to_excel(excel_writer=excel_writer, sheet_name=sheet_name)

    def read_excel(self, path, sheet_name="Scorecard"):
        """
        Import scorecard from excel file.

        Parameters
        ----------
        path : str
            Path to xls or xlsx file.

        sheet_name : str, (default="Scorecard")
            Name of the sheet which will contain the scorecard.
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string.")

        if not isinstance(sheet_name, str):
            raise TypeError("sheet_name must be a string.")

        self._table = pd.read_excel(io=path, sheet_name=sheet_name)

    def load(self, pickle_path):
        """
        Load scorecard object from pickle.

        Parameters
        ----------
        pickle_path : str
            Path to pickle object.
        """
        if not isinstance(pickle_path, str):
            raise TypeError("pickle_path must be a string.")

        with open(pickle_path, "rb") as input_file:
            scorecard = pickle.load(input_file)

        self.__dict__ = copy.deepcopy(scorecard.__dict__)

    def _save(self, pickle_path):
        """
        Save scorecard object to pickle.
        We recommend use the function export to get the scorecard. This is
        the reason for definning this function as private.

        Parameters
        ----------
        pickle_path : str
            Path to pickle object.
        """
        if not isinstance(pickle_path, str):
            raise TypeError("pickle_path must be a string.")

        with open(pickle_path, "wb") as output_file:
            pickle.dump(self, output_file)

    def export(self, path, file_name="Scorecard", sheet_name="Scorecard"):
        """
        Export scorecard to excel and the scorecard Class to pickle file.
        We recomend the use of this procedure to save the scorecards.
        Additionally, this allows keeping track of which model was
        used to built the scorecard.

        Parameters
        ----------
        path : str
            Path where files will be saved.

        file_name : str
            Name of the .xlsx and pickle files.

        sheet_name : str
            Sheet name of the excel where the scorecard will be saved.

        """
        if not isinstance(path, str):
            raise TypeError("path must be a string.")
        if not isinstance(file_name, str):
            raise TypeError("file_name must be a string.")
        if not isinstance(sheet_name, str):
            raise TypeError("sheet_name must be a string.")

        if not os.path.exists(path):
            os.makedirs(path)
        system_os = platform.system()
        if system_os == "Windows":
            file_path_excel = path + '\\' + file_name + '.xlsx'
            file_path = path + '\\' + file_name
        else:
            file_path_excel = path + '/' + file_name + '.xlsx'
            file_path = path + '/' + file_name
        self._to_excel(excel_writer=file_path_excel, sheet_name=sheet_name)
        self._save(file_path)

    def _fit(self):
        """
        Build scorecard.

        Returns
        -------
        self : object
        """
        if not isinstance(self.model, GRMlabModelClassification):
            raise TypeError()

        if not isinstance(self.grouping, (Bivariate, OptimalGrouping)):
            raise TypeError()

        if hasattr(self.grouping, "optimalgrouping"):
            self._optimalgrouping = self.grouping.optimalgrouping
        else:
            self._optimalgrouping = self.grouping

        # build scorecard
        if hasattr(self.model.estimator, "coef_"):
            coefs = self.model.estimator.coef_
            is_coef_positive = False
        elif hasattr(self.model.estimator, "feature_importances_"):
            coefs = self.model.estimator.feature_importances_
            is_coef_positive = True
        else:
            raise RuntimeError("The estimator does not expose 'coef_' "
                               "or 'feature_importances_' attributes.")

        coefs = coefs.ravel()

        feature_names = self.model.get_support(output_names=True)

        binnings = []
        min_score = []
        max_score = []
        scores = []

        for i, feature in enumerate(feature_names):
            binning = self._optimalgrouping.variable_binning_table(feature)
            binning = binning.drop("Total")
            binning["Name"] = feature
            records = binning["Event count"] + binning["Non-event count"]
            binning["Records (%)"] = records / records.sum()

            woe = binning["WoE"].values
            score = coefs[i] * woe
            scores.append(score)

            min_score.append(min(score))
            max_score.append(max(score))
            binnings.append(binning)

        table = pd.concat(binnings)

        # scale / normalize
        max_points = np.sum(max_score)
        min_points = np.sum(min_score)

        scaling = 100.0 / (min_points - max_points)

        if is_coef_positive:
            scaling *= - 1

        points = []
        for score in scores:
            scaled = score * scaling
            points.extend(abs(min(scaled)) + scaled)

        table["Score"] = points
        table["Group"] = table.index

        cols = ["Name", "Splits", "Group", "WoE", "Records (%)", "Event count",
                "Non-event count", "Default rate", "Score"]

        self._table = table[cols].reset_index(drop=True)

        self._is_fitted = True

    def _transform(self, data, feature):
        """
        Transform the feature column into WoEs with the optimal binning
        contained in the scorecard class.
        """
        feature_id = self._optimalgrouping._detect_name_id(feature)
        optbin = self._optimalgrouping._variables_information[feature_id]

        return optbin.transform(data)
