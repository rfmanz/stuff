import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from smart_open import open
from sklearn.base import TransformerMixin, BaseEstimator
from ml4risk.data_preparation.woe import WOE_Transform, get_monotone_dir
from collections import defaultdict as ddict


class WOEImputer(TransformerMixin, BaseEstimator):
    """
    Imputer based on WOE Transformation for Numerical features

    Currently does not support special (categorical) values

    For each column, impute the missing values with
    - the mean of the bin closest to missing, in terms of WOE
    - if bin contains np.inf/-np.inf, impute with min(bin) or max(bin) respectively

    following the SimpleImputer Example
    https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/impute/_base.py#L120
    """

    def __init__(
        self,
        impute_method="midpoint",
        woe_method="tree",
        num_bin_start=40,
        min_samples_leaf=500,
        state_dict_path=None,
        woe=None
    ):
        """
        Parameters
        ----------
        impute_method

        woe_method

        num_bin_start

        min_samples_leaf

        state_dict_path

        Attributes
        ----------
        impute_values_

        """
        print("using imputer v2")
        if woe is None:
            self.woe = WOE_Transform(
                method=woe_method,
                num_bin_start=num_bin_start,
                min_samples_leaf=min_samples_leaf,
                min_iv=-np.inf,
            )
        else:
            self.woe = woe

        self.impute_method = impute_method

        if state_dict_path is not None:
            with open(state_dict_path, "rb") as f:
                state_dict = pkl.load(f)
                self.impute_values_ = state_dict["impute_values"]
                self.impute_method = state_dict["impute_method"]
                self.woe_dict = state_dict["woe_dict"]

    def _validate_input(self, X, y):
        """
        assert targets are:
        * binary
        * non-missing
        """
        assert not y.isna().any()

        tvals = sorted(y.unique())
        assert np.array_equal(tvals, [0, 1])

    def fit(self, X, y, weights=None, min_dict=None, max_dict=None):
        """
        Parameter
        ---------
        X : pd.DataFrame
            columns to be imputed
        y : pd.Series - Binary
            target column to compute WOE
        """
        self._validate_input(X, y)

        self.woe.fit(X, y.astype(int), Y_weight=weights, display=-1)
        self.woe_dict = self.woe.woe_dict()
        self.impute_values_ = {}
        self.failed_to_impute = []
        
        if min_dict is None:
            min_dict = {}
        if max_dict is None:
            max_dict = {}
        self.min_dict = ddict(lambda: None, min_dict)
        self.max_dict = ddict(lambda: None, max_dict)

        for f in tqdm(X.columns):
            if f not in self.woe_dict:
                self.failed_to_impute.append(f)
                continue
            val = get_missing_impute_value_v2(self.woe_dict[f],
                                           method=self.impute_method, 
                                           min_val=self.min_dict[f],
                                           max_val=self.max_dict[f])
            if val is not None:
                self.impute_values_[f] = val
        print("failed to impute following features: ")
        print(self.failed_to_impute)
        

    def transform(self, X, y=None, inplace=False):
        """
        Parameter
        ---------
        X : pd.DataFrame
            columns to be imputed
        y : pd.Series - Binary
            target column to compute WOE

        Returns
        -------
        X_ : X_ with missing values imputed
        """

        X_ = X if inplace else X.copy(deep=True)

        if not hasattr(self, "impute_values_"):
            raise ValueError(
                "Transformer not fitted. Either call fit or load previously computed woe_dict before transforming data."
            )

        for f in tqdm(X.columns):
            ## add passing if no missing

            if f in self.impute_values_:
                X_[f].fillna(self.impute_values_[f], inplace=inplace)

        return X_

    def save_state_dict(self, path):
        """
        save self.impute_values_ dict to provided path

        can be loaded for transformations
        """
        if not hasattr(self, "impute_values_"):
            raise ValueError(
                "Transformer not fitted. Either call fit or load previously computed woe_dict before transforming data."
            )

        state_dict = {
            "impute_values": self.impute_values_,
            "impute_method": self.impute_method,
            "woe_dict": self.woe_dict,
        }
        with open(path, "wb") as f:
            pkl.dump(state_dict, f)


def get_missing_impute_value(woe_table, min_val=None, max_val=None, method="closest_boundary"):
    """if missing in woe_table.index
    return the imputed value of the bin closest to missing in terms of WOE
    """
    assert method in ["midpoint", "closest_boundary"]

    if "missing" not in woe_table.index:
        return None

    missing_woe = woe_table.loc["missing", "woe"]
    woe_table["distance"] = (woe_table["woe"] - missing_woe).abs()
    woe_table = woe_table.loc[woe_table.index != "missing"]
    closest_bin = woe_table.sort_values("distance").head(1)
    max_woe = woe_table["woe"].max()
    min_woe = woe_table["woe"].min()
    if not min_val:
        min_val = woe_table["min"].min()
    if not max_val:
        max_val = woe_table["max"].max()
    
    mc_dict = get_monotone_dir({"tmp": woe_table})
    # pos if 1, neg if -1, None if only a single non-nan value
    mc_dir = mc_dict["tmp"] if "tmp" in mc_dict else None

    if method in ["midpoint"]:
        mapping_fn = {"midpoint": "mean"}
        if (max_woe < missing_woe) & (mc_dir == 1):
            result = max_val + 10
        elif (max_woe < missing_woe) & (mc_dir == -1):
            result = min_val - 10        
        elif (min_val > missing_woe) & (mc_dir == 1):
            result = min_val - 10   
        elif (min_val > missing_woe) & (mc_dir == -1):
            result = max_val + 10
        else:
            result = getattr(closest_bin[["min", "max"]].values, mapping_fn[method])()
    elif method in ["closest_boundary"]:       
        if (max_woe < missing_woe) & (mc_dir == 1):
            result = max_val + 10
        elif (max_woe < missing_woe) & (mc_dir == -1):
            result = min_val - 10        
        elif (min_val > missing_woe) & (mc_dir == 1):
            result = min_val - 10   
        elif (min_val > missing_woe) & (mc_dir == -1):
            result = max_val + 10
        elif (closest_bin["woe"].item() <= missing_woe) & (mc_dir is None):
            result = closest_bin["max"].item()
        elif (closest_bin["woe"].item() > missing_woe) & (mc_dir is None):
            result = closest_bin["min"].item()
        elif (closest_bin["woe"].item() <= missing_woe) & (mc_dir == 1):
            result = closest_bin["max"].item()
        elif (closest_bin["woe"].item() <= missing_woe) & (mc_dir == -1):
            result = closest_bin["min"].item()
        elif (closest_bin["woe"].item() > missing_woe) & (mc_dir == 1):
            result = closest_bin["min"].item()
        elif (closest_bin["woe"].item() > missing_woe) & (mc_dir == -1):
            result = closest_bin["max"].item()
        else:
            raise ValueError("invalid case...")

    return result



def get_missing_impute_value_v2(woe_table, min_val=None, max_val=None, method="midpoint"):
    """
    Update: limit encoded value to observed min/max
    if missing in woe_table.index
    return the imputed value of the bin closest to missing in terms of WOE
    """
    assert method in ["midpoint", "closest_boundary"]
    if min_val:
        assert min_val <= woe_table["min"].min()
    if max_val:
        assert max_val >= woe_table["max"].max()

    if "missing" not in woe_table.index:
        return None

    missing_woe = woe_table.loc["missing", "woe"]
    woe_table["distance"] = (woe_table["woe"] - missing_woe).abs()
    woe_table = woe_table.loc[woe_table.index != "missing"]
    closest_bin = woe_table.sort_values("distance").head(1)
    max_woe = woe_table["woe"].max()
    min_woe = woe_table["woe"].min()
    
    mc_dict = get_monotone_dir({"tmp": woe_table})
    # pos if 1, neg if -1, None if only a single non-nan value
    mc_dir = mc_dict["tmp"] if "tmp" in mc_dict else None
    if method in ["midpoint"]:
        mapping_fn = {"midpoint": "mean"}
        if (max_woe < missing_woe) & (mc_dir == 1):
            result = woe_table["max"].max()
        elif (max_woe < missing_woe) & (mc_dir == -1):
            result = woe_table["min"].min()  
        elif (min_woe > missing_woe) & (mc_dir == 1):
            result = woe_table["min"].min()   
        elif (min_woe > missing_woe) & (mc_dir == -1):
            result = woe_table["max"].max()
        else:
            result = getattr(closest_bin[["min", "max"]].values, mapping_fn[method])()
    elif method in ["closest_boundary"]:       
        if (max_woe < missing_woe) & (mc_dir == 1):
            result = woe_table["max"].max()
        elif (max_woe < missing_woe) & (mc_dir == -1):
            result = woe_table["min"].min()        
        elif (min_woe > missing_woe) & (mc_dir == 1):
            result = woe_table["min"].min()  
        elif (min_woe > missing_woe) & (mc_dir == -1):
            result = woe_table["max"].max()
        elif (closest_bin["woe"].item() <= missing_woe) & (mc_dir is None):
            result = closest_bin["max"].item()
        elif (closest_bin["woe"].item() > missing_woe) & (mc_dir is None):
            result = closest_bin["min"].item()
        elif (closest_bin["woe"].item() <= missing_woe) & (mc_dir == 1):
            result = closest_bin["max"].item()
        elif (closest_bin["woe"].item() <= missing_woe) & (mc_dir == -1):
            result = closest_bin["min"].item()
        elif (closest_bin["woe"].item() > missing_woe) & (mc_dir == 1):
            result = closest_bin["min"].item()
        elif (closest_bin["woe"].item() > missing_woe) & (mc_dir == -1):
            result = closest_bin["max"].item()
        else:
            raise ValueError("invalid case...")

    return result