"""
GRMlab classification model
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import pandas as pd

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score

from ..core.exceptions import NotFittedException
from .base import GRMlabModel
from .metrics import auc_pr
from .metrics import auc_roc
from .metrics import gini
from .metrics import imbalanced_classification_report


class GRMlabModelClassification(GRMlabModel):
    """
    GRMlab model for classification models.
    """
    def metrics(self, X, y, binary=False, target_names=None,
                output_dict=False):
        """
        Calculate classification metrics.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            The training input samples.

        y : array-like, shape = (n_samples)
            The target values.

        binary : bool (default=False)
            Whether to report only the positive class, i.e. class = 1 when
            target is binary. Otherwise the report returns metrics for class 0
            and 1.

        target_names : list of strings, (default=None)
            Optional display column names matching the labels (same order).

        output_dict : bool (default=False)
            If True, return output as dict, otherwise return pandas.DataFrame.

        Returns
        -------
        report : dict or pandas.DataFrame
        """
        if not self._is_estimator_fitted:
            raise NotFittedException(self)

        y_pred = self.predict(X)

        imetrics = imbalanced_classification_report(
            y, y_pred, binary=binary, target_names=target_names,
            output_dict=output_dict)

        if binary:
            y_pred_proba = self.predict_proba(X)[:, 1]

            if output_dict is False:
                new_metrics = imetrics[imetrics.columns[0]].to_dict()
            else:
                new_metrics = imetrics

            new_metrics["auc_roc"] = auc_roc(y, y_pred_proba)
            new_metrics["auc_pr"] = auc_pr(y, y_pred_proba)
            new_metrics["cohen_kappa"] = cohen_kappa_score(y, y_pred)
            new_metrics["f1_score"] = f1_score(y, y_pred)
            new_metrics["gini"] = gini(y, y_pred_proba)

            if output_dict is False:
                return pd.DataFrame.from_dict(
                    new_metrics, orient='index').sort_index()
            else:
                return new_metrics
        else:
            return imetrics
