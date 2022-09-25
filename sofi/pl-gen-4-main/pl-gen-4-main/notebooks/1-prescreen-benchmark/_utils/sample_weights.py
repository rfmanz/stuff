import pandas as pd
from typing import Dict
import warnings


def get_ratio_weights(df, groupby_col, normalize_by, weight_col=None):
    """
    get sample weight by ratio, of the groups in weights
    then normalize w.r.t the <normalize_by> group to have unit weight
    
    # or should i normalize by the total weight of the dataset?
    """
    assert (df[groupby_col] == normalize_by).any()
    # compute updated weight
    if weight_col:
        weights = 1 / (df[[weight_col, groupby_col]]
                       .groupby(groupby_col)[weight_col].sum())   
    else:
        weights = 1 / df[groupby_col].value_counts(normalize=True)
    
    weights = weights / weights.loc[normalize_by] 
    return weights.to_dict()
    

def get_naive_sample_weight(df, groupby_col, weights):
    assert sorted(df[groupby_col].unique().tolist()) == sorted(list(weights.keys()))
    # not ratio
    series = pd.Series(index=df.index, dtype=float)
    for k in weights:
        series.loc[df[groupby_col]==k] = weights[k]
    if series.isna().any():
        # should never enter here thou because of assert above
        warnings.warn("series contain missing. weights provided may not be comprehensive")    
    return series


def get_sample_weight(df: pd.DataFrame, groupby_col: str, 
                      weights: Dict[str,float]={}, 
                      ratio=False, normalize_by=None, weight_col=None,
                      return_weights=False):
    """
    generate sample weight based on provided weights and column value of col
    
    if ratio is True, normalize the segment weights to obtain the target ratio
        get sample weight by ratio, of the groups in weights
        then normalize w.r.t the <normalize_by> group to have unit weight
        
    @params df: pd.DataFrame
    @params groupby_col: str 
            column to stratize by for weights
    @params weights: dict[str, float] 
            if provided, simply give rows corresponding
            weights indicated in the dict
    @params ratio: bool
            whether automatically find weights to produce intended weights
    @params normalize_by: str
            which groupby_col value to be set as the unit weight
    @params weight_col: str
            if provided, take consideration of current weights (from other criteria)
            then generate scaling sample weights such that
            provided weights * sample weights would guarante final weight ratio
    @params return_weights: bool
            whether to obtain the weight scaler as a record
    """
    # must provide weights, or provide ratio and normalize_by value
    assert((len(weights) > 0) or (ratio and normalize_by))
    if ratio:
        assert normalize_by in ratio
        weights = get_ratio_weights(df, groupby_col, normalize_by, weight_col=weight_col)
        sample_weights = get_naive_sample_weight(df, groupby_col, weights)
    else:
        sample_weights = get_naive_sample_weight(df, groupby_col, weights)
        
    if return_weights:
        return sample_weights, weights
    return sample_weights