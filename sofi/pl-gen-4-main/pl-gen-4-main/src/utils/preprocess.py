import pandas as pd
import numpy as np
import sklearn

### Outlier Detection

# table with split_feature, lower, upper
# to do that, we need split_feature, s1, s2, sk-1, sk

# manual
# clip_tbl = get_lgbm_split_caps(clf.booster_)
# display(X.describe().loc[["mean", "std", "min", "max"]].round(2))
# X = clip_df(X, clip_tbl.index, clip_tbl)
# display(X.describe().loc[["mean", "std", "min", "max"]].round(2))


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


def clip_df(df, features, clip_tbl):
    """
    clip df by features with "lower" and "upper" values in clip_tbl
    """
    for f in features:
        lower = clip_tbl.loc[f]["lower"]
        upper = clip_tbl.loc[f]["upper"]
        df[f].clip(lower, upper, inplace=True)
    return df

### Skewness

from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from tqdm import tqdm

# bulk transform
# X_transformed = power_transform(X, method='yeo-johnson', standardize=True)

# manual
# X, tfmrs = append_transform_cols(X, features)

# def append_transform_cols_(df, features):
#     tfmr = PowerTransformer(method='yeo-johnson', standardize=True)
#     tf_cols = [f"{f}_xf" for f in features]
#     tfmr.fit(df[features])
    
#     result_df = df.copy()
#     result_df[tf_cols] = tfmr.transform(result_df[features])
#     return result_df, tfmr


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