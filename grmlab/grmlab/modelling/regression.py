"""
GRMlab regression model.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

from ..core.exceptions import NotFittedException
from .base import GRMlabModel
from .metrics import regression_report


class GRMlabModelRegression(GRMlabModel):
    """
    GRMlab model for regression problems.
    """
    def metrics(self, X, y, output_dict=False):
        """
        Calculate regression metrics.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            The training input samples.

        y : array-like, shape = (n_samples)
            The target values.

        output_dict : bool (default=False)
            If True, return output as dict, otherwise return pandas.DataFrame.

        Returns
        -------
        report : dict or pandas.DataFrame
        """
        if not self._is_estimator_fitted:
            raise NotFittedException(self)

        y_pred = self.predict(X)

        return regression_report(y, y_pred, output_dict=output_dict)
