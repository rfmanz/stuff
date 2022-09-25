import pandas as pd
import ast
import numpy as np
from tqdm import tqdm

def encode_special_to_missing(df, df_dict, col_name):
    # encode special value to missing
    df_ = df.copy()
    for col in tqdm(col_name):
        if col not in df_dict.field_name.tolist(): # if not Experian attributes
            pass
        elif df_dict[df_dict.field_name == col].categorical.isnull().iloc[0] :    # if no special values
            pass
        else: # if Experian attributes with special value, impute special value to missing
            special_val = ast.literal_eval(df_dict[df_dict.field_name == col].categorical.iloc[0])
            special_val = [int(i) for i in special_val]
            df_.loc[df_[col].isin(special_val), col] = np.nan
    return df_

def get_missing_impute_value(woe_table):
    """ if missing in woe_table.index
    return the mean of the bin closest to missing in terms of WOE
    """
    if "missing" not in woe_table.index:
        return None
    
    woe_table = woe_table.sort_values("woe")
    woe_table.sort_values("woe")
    woe_table["distance"] = (woe_table["woe"] - woe_table.loc["missing", "woe"]).abs()
    woe_table = woe_table.loc[woe_table.index != "missing"]
    closest_bin = woe_table.sort_values("distance").head(1)
    if closest_bin.min == -np.inf:
        return closest_bin.max
    elif closest_bin.max == np.inf:
        return closest_bin.min
    return closest_bin[["min", "max"]].values.mean()