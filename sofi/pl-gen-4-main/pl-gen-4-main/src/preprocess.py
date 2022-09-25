import sys, json, os, ast, copy
import numpy as np
import pandas as pd
import pickle as pkl
import lightgbm as lgb

from smart_open import open
from tqdm import tqdm
from typing import Dict
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from tqdm import tqdm
from typing import List


from .utils.sample_weights import get_sample_weight
import src.imputer as ip


class Preprocess:
    
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.no_special_cols = []
        self.manual_treatment_required = []
    
    def encode_cat_to_missing(self, df: pd.DataFrame, 
                               features: list,
                               data_dict: pd.DataFrame):
        assert pd.Series(features).isin(df.columns).all(), "some features not present"

        for col in tqdm(features):
            try:
                cats = data_dict[data_dict.field_name == col].categorical.iloc[0]
                if isinstance(cats, float):  # nan
                    self.no_special_cols.append(col)
                    continue
                special_val = ast.literal_eval(cats)
                special_val = [int(i) for i in special_val]
                df[col] = cat_2_nan_series(df[col], special_val) 
            except:
                self.manual_treatment_required.append(col)
        return df, self.no_special_cols
    
    def apply_sample_weight(self, df, groupby_col, weights, existing_weights_col=None):
        print(f"""
        added columns:
            weight: training sample_weight scaled using provided weights by ri_source
                {existing_weights_col} * weight_sample
        """) 
        df["weight_sample"], weights = get_sample_weight(df, 
                groupby_col, weights, return_weights=True)
        if existing_weights_col is not None:
            df["weight"] = df["weight_sample"] * df[existing_weights_col]
        else:
            df["weight"] = df["weight_sample"]
        return df, weights
    
    def dropping_indeterminate(self, df, indeterminate_col):
        return df[~df[indeterminate_col]]
    
    def transform(self, df, features, weights, groupby_col="ri_source",
                  drop_indeterminate=None, existing_weights_col=None):
        df, self.no_special_cols = self.encode_cat_to_missing(df, features,
                                                              self.data_dict)
        df, self.sample_weight = self.apply_sample_weight(df, groupby_col, weights, 
                                                          existing_weights_col=existing_weights_col)
        
        if drop_indeterminate:
            print(f"dropping indeterminate col: {drop_indeterminate}")
            df = self.dropping_indeterminate(df, drop_indeterminate)
        return df
    
        
######################################################
#  encode columns to missing - time consuming so far
######################################################

def cat_2_nan_series(series, categorical_list):
    """ given series and a list of catergorical values
    
    replace the categorical occurances to nan
    """
    if len(categorical_list) == 0:
        return series
    mapper = dict((k, np.nan) for k in categorical_list)
    return series.replace(mapper)




######################################################
#       to get monotonic directions
######################################################

def get_monotone_dir(woe_dict):
    result = {}
    for k in woe_dict:
        tbl = woe_dict[k]
        if len(tbl) < 2:
            print(k, len(tbl))
        elif tbl.iloc[0]["woe"] < tbl.iloc[1]["woe"]:
            direction = 1
        else:
            direction = -1
        
        result[k] = direction
    return result

######################################################
#  
######################################################



######################################################
#              data clipping
######################################################


def get_thresh(df):
    ss = df.nsmallest(2, "threshold")
    sl = df.nlargest(2, "threshold")
    result = pd.concat([ss, sl]).sort_values("threshold")
    result["bin_tags"] = ["s1", "s2", "sk-1", "sk"]
    return result
    
def get_caps(tbl):
    # get s1, s2, sk-1, sk
    tbl_ = tbl.groupby("split_feature").apply(get_thresh).reset_index(drop=True)
    index = pd.MultiIndex.from_frame(tbl_[["split_feature", "bin_tags"]])
    tbl_.index = index
    _ = tbl_.pop("split_feature")
    _ = tbl_.pop("bin_tags")
    tbl_ = tbl_.unstack()   
    
    # formatting before extracting values
    tbl_.columns = map(lambda t: t[1], tbl_.columns)
    
    # get lower and upper
    tbl_["lower"] = 2*tbl_["s1"] - tbl_["s2"]
    tbl_["upper"] = 2*tbl_["sk"] - tbl_["sk-1"]
    return tbl_



def get_lgbm_split_caps(lgbm_booster):
    feature_split_tbl = lgbm_booster.trees_to_dataframe()
    tbl = feature_split_tbl[["split_feature", "threshold"]].drop_duplicates()
    tbl = tbl.sort_values(["split_feature", "threshold"]).reset_index(drop=True).dropna()
    tbl = get_caps(tbl)
    return tbl


def get_clipping_tbl_lgbm(df: pd.DataFrame, 
                          features: List[str],
                          target: str,
                          weight_col: str = None,
                          params=None) -> pd.DataFrame:
    """
    given dataframe, features, and target
    train an single tree lgbm on the df to produce poc cutoff table
    for clipping purposes. 
    
    only used for feature selection at poc stage.
    it is recommended to clip and process each features manually
    after it is selected.
    
    returns clip_tbl and columns reguire manual attention
    """
    tbl_row = []
    irreg_row = []
    if params is None:
        params = {'boosting_type': 'gbdt',
                  'learning_rate': 0.05,
                  'n_estimators': 1,
                  'min_data_in_leaf': 300,
                  'verbose':-1,
                 }

    for ft in tqdm(features):
        clf = lgb.LGBMClassifier(**params)
        weights = df[weight_col] if weight_col else None
        clf.fit(df[[ft]], df[target],
                sample_weight=weights)
        try:
            tbl_row.append(get_lgbm_split_caps(clf.booster_))
        except:
            irreg_row.append(ft)
    clip_tbl = pd.concat(tbl_row, axis=0)
    return clip_tbl, irreg_row


def clip_df(df, features, clip_tbl, method="min_max"):
    """
    clip df by features with "lower" and "upper" values in clip_tbl
    """
    assert(method in ["min_max", "drop_outliers"])
    for f in tqdm(features):
        if f in df.columns:
            if method == "min_max":
                lower = clip_tbl.loc[f]["s1"]
                upper = clip_tbl.loc[f]["sk-1"]
            elif method == "drop_outliers":
                lower = clip_tbl.loc[f]["s2"]
                upper = clip_tbl.loc[f]["sk-1"]
            df[f].clip(lower-1, upper+1, inplace=True)
    return df



######################################################
#        power transformation
######################################################


def append_transform_cols(df, features, method='yeo-johnson', suffix="_xf"):
    """ the iterative method """
    tfmrs = {}
    df_ = df.copy()
    
    for f in tqdm(features):
        x = df[f].values.reshape(-1,1)
        if method == 'yeo-johnson':
            tfmr = PowerTransformer(method=method, standardize=True)
        elif method == "quantile":
            tfmr = QuantileTransformer(output_distribution="normal")
        tfmr.fit(x)
        df_[f"{f}{suffix}"] = tfmr.transform(x)
        tfmrs[f] = tfmr
    
    return df_, tfmrs


def get_initial_bias(target, weight=None):
    """
    compute initial bias for imbalanced binary data
    
    given target and weight array, produce initial bias
    
    ref: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias
    """
    import numpy as np
    import math
    
    if weight is None:
        weight = np.ones_like(target)
    assert(sorted(list(np.unique(target))) == [0,1])
    pos_w = weight[target.astype(bool)].sum()
    neg_w = weight[~target.astype(bool)].sum()
    b = math.log(pos_w/neg_w)
    return b



######################################################
#        DL Preprocessor
######################################################

