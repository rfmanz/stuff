import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Sequence
from numbers import Number
from collections import defaultdict


class DataWrangler:
    def __init__(self):
        return

    def clip_below(self, series: pd.Series, value: Number) -> pd.Series:
        return series.clip(lower=value)

    def clip_above(self, series: pd.Series, value: Number) -> pd.Series:
        return series.clip(upper=value)

    def fillna(self, series: pd.Series, value: Number) -> pd.Series:
        return series.fillna(value, inplace=False)

    def get_finite_not_nan(self, series: pd.Series) -> pd.Series:
        return series.loc[np.isfinite(series) & ~series.isna()]

    def quantile(self, series: pd.Series, value: Number) -> Number:
        """
        compute quantile ignoring [np.nan, -np.inf, np.inf]
        """
        data = self.get_finite_not_nan(series)
        return data.quantile(value)

    def clip(
        self, series: pd.Series, lower: Number = -np.inf, upper: Number = np.inf
    ) -> pd.Series:
        assert lower <= upper
        return series.clip(lower=lower, upper=upper)

    def replace(self, series: pd.Series, mapping: dict) -> pd.Series:
        return series.replace(mapping)

    def clean_feature(
        self,
        series: pd.Series,
        lower: Number = -np.inf,
        upper: Number = np.inf,
        fillna: Number = None,
        mapping: dict = {},
    ) -> pd.Series:
        """
        RIGHT NOW BUGGY AS CLIP TOP IS NOT WORKING
        
        clean feature in the following order:
        - replace
        - clip
        - fillna
        """
        result = series.copy()
        if len(mapping) > 0:
            result = self.replace(result, mapping)
        if not np.isfinite(lower) or not np.isfinite(upper):
            result = self.clip(result, lower, upper)
        if fillna is not None:
            result = self.fillna(result, fillna)
        return result
    
    def describe(self, series):
        pass

    def hist(self, series: pd.Series, bins=10, **kwargs):
        """
        plot distribution/histogram as if the -inf, inf and nan are removed

        takes in series and all params to plt.hist()
        """
        fig, ax = plt.subplots()
        data = self.get_finite_not_nan(series)
        data.hist(bins=bins, **kwargs)
        return fig, ax

    def encode_special(self, df:pd.DataFrame,
                    feature: str,
                    interval: pd.Interval, 
                    encode_special: bool):
        """
        Replace special values (beyond the provided interval inclusive) with NaN.
        If encode_special set to True, int encode them to another column.
        """
        # set up variables
        k = feature
        v = interval
        encode = encode_special
        df = df[feature].copy(deep=True).to_frame()
        cname = k + '_encoded'

        if isinstance(v, pd.Interval):
            is_default = ~df[k].between(v.left, v.right) & ~df[k].isna()
        elif isinstance(v, list):
            is_default = df[k].isin(k)
        else:
            raise RuntimeError('Data type {} not supported'.format(str(type(v))))

        if ~is_default.isna().all():
            if encode:
                df.loc[is_default, cname] = is_default * df[k]
            df.loc[is_default, k] = np.nan #set default values to NaN
        
        feature_col = df[feature]
        
        encoded_col = None
        if encode:
            encoded_col = df[cname]
        return feature_col, encoded_col