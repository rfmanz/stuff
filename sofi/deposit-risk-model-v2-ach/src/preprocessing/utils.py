import pandas as pd
import numpy as np


def preprocess(df):

    f = "vantage_score"
    # encoding
    feature_values, encoded_values = encode_special(
        df, f, pd.Interval(300, 850), False
    )
    df[f] = feature_values
    if encoded_values is not None:
        df[f + "_encoded"] = encoded_values

    f = "bcc7120"
    # encoding
    feature_values, encoded_values = encode_special(df, f, pd.Interval(0, 990), True)
    df[f] = feature_values
    if encoded_values is not None:
        encode_col = f + "_encoded"
        df[encode_col] = encoded_values
    # fillna
    df[f] = df[f].fillna(-1)
    df[encode_col] = df[encode_col].fillna(-1)

    f = "fico_score"
    # encoding
    feature_values, encoded_values = encode_special(df, f, pd.Interval(0, 850), True)
    df[f] = feature_values
    if encoded_values is not None:
        encode_col = f + "_encoded"
        df[encode_col] = encoded_values

    df["first_deposit_amount"] = df["first_deposit_amount"].clip(0, 20000).fillna(-1)
    df["credit_card_loan_amount"] = (
        df["credit_card_loan_amount"].clip(0, 100000).fillna(-1)
    )
    df["plaid_max_avail_bal"] = df["plaid_max_avail_bal"].clip(-100000, 100000)
    df["total_tradelines_open"] = df["total_tradelines_open"].clip(0, 31).fillna(-1)
    df["plaid_days_since_first_link"] = (
        df["plaid_days_since_first_link"].clip(0, 69).fillna(-1)
    )
    df["plaid_min_avail_bal"] = df["plaid_min_avail_bal"].clip(-100000, 100000)
    df["nr_transactions_per_day"] = df["nr_transactions_per_day"].clip(0, 5).fillna(0)
    df["sum_deposits_10d"] = df["sum_deposits_10d"].clip(0, 25000).fillna(0)
    df["total_outstanding_balance"] = df["total_outstanding_balance"].clip(0, 500000)
    df["rolling_trns_as_pct_of_bal"] = df["rolling_trns_as_pct_of_bal"].clip(-750, 750)
    df["phone_risk_score"] = df["phone_risk_score"].clip(0, 1)
    df["email_risk_score"] = df["email_risk_score"].clip(0, 1)
    df["deposits_ratio"] = df["deposits_ratio"].fillna(-1)
    df["fraud_score_1"] = df["fraud_score_1"].clip(0, 1)
    df["mean_deposits_10d_div_mean_deposits"] = (
        df["mean_deposits_10d_div_mean_deposits"].clip(0, 100).fillna(-100)
    )
    df["fraud_score_2"] = df["fraud_score_2"].clip(0, 1)
    df["nr_past_deposits"] = df["nr_past_deposits"].clip(0, 200)
    df["address_risk_score"] = df["address_risk_score"].clip(0, 1)

    # check up on vantage_score
    # nr_transactions_30d_div_nr_past_transactions
    # should clip by (0, 1)
    df["nr_transactions_30d_div_nr_past_transactions"] = (
        df["nr_transactions_30d_div_nr_past_transactions"].clip(0, 2).fillna(-1)
    )
    # max_deposits_10d_div_mean_account_balance_30d
    df["max_deposits_10d_div_mean_account_balance_30d"] = (
        df["max_deposits_10d_div_mean_account_balance_30d"].clip(0, 10).fillna(-1)
    )
    return df


def encode_special(df:pd.DataFrame,
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