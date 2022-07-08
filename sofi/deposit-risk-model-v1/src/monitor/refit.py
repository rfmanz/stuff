"""
Training scripts.
"""

import argparse
import os
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


def pickleobj(obj, path):
    """
    Pickle an object.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        

def get_timestamp_str():
    """
    Return string of current timestamp.
    """
    return str(int(time.time()))


targets = [{'target_col_name': 'target'}]

from collections import defaultdict

def encode_defaults(df, default_values):
    """Replace default values with NaN, int encode them"""
    default_encoded_cols = []
    for k, (v, encode) in default_values.items():
        cname = k + '_default_encoded'

        if isinstance(v, pd.Interval):
            is_default = ~df[k].between(v.left, v.right) & ~df[k].isna()
        elif isinstance(v, list):
            is_default = df[k].isin(k)
        else:
            raise RuntimeError('Data type {} not supported'.format(str(type(v))))
        
        if ~is_default.isna().all():
            if encode:
                default_encoded_cols.append(cname)
                df.loc[is_default, cname] = is_default * df[k]
            df.loc[is_default, k] = np.nan #set default values to NaN
        
    return df, default_encoded_cols


cols_raw = ['first_deposit_amount', 'vantage_score', 'bcc7120', 'email_risk_score', 
            'fraud_score_2', 'name_email_correlation', 'transaction_as_pct_of_balance', 
            'mean_account_balance_30d', 'phone_risk_score',  # giact_time_since_last_link removed 
            'name_address_correlation', 'all8220', 'lag_acc_open_first_transaction', 
            'dollar_val_dd', 'all7120', 'sum_deposits_10d', 'nr_past_transactions', 
            'total_tradelines_open', 'education_loan_amount', 'address_risk_score',
            'iqt9415', 'max_withdrawals_30d', 'iln5520', 'max_deposits_30d', 
            'pct_returned_deposits', 'giact_nr_decline', 'nr_direct_deposits', 
            'time_since_last_transaction', 'bal_ratio', 'name_phone_correlation', 
            'giact_nr_other', 'dollar_val_returns', 'nr_trans_ratio', 'iqt9413', 
            'dollar_val_returns_3d', 'nr_returns_30d', 'credit_card_loan_amount',
            'fraud_score_1', 'age_money_account']

cols_raw_ni = ['transaction_code']

cols_encoded = ['transaction_code_encoded', 'all7120_default_encoded', 'bcc7120_default_encoded']

features = cols_raw + cols_encoded

default_values = {
    'vantage_score': [pd.Interval(300, 850), False],
    'all7120': [pd.Interval(0, 990), True],
    'all8220': [pd.Interval(0, 9990), False],
    'bcc7120': [pd.Interval(0, 990), True],
    'iln5520': [pd.Interval(0, 999999990), False],
    'iqt9413': [pd.Interval(0, 90), False]
}

def preprocess(df):
    """
    Code to preprocess model.
    """
    # mapping from transaction code to integer
    tcode_map = defaultdict(int,
                            {'POSDW': 1,
                             'ACHDD': 2,
                             'ACHDWIN': 3,
                             'ACHDDIN': 4,
                             'ACHDW': 5,
                             'ACHINDD': 6,
                             'DDATMREFUND': 7,
                             'DWATM': 8,
                             'DDRAFNEW':9,
                             'DDCK': 10})

    df['transaction_code_encoded'] = df['transaction_code'].map(tcode_map)
    
    # clip to handle infinite values and outliers
    df['transaction_as_pct_of_balance'] = df['transaction_as_pct_of_balance'].clip(-750, 750)
    df['bal_ratio'] = df['bal_ratio'].clip(-750, 750)
    df['pct_returned_deposits'] = df['pct_returned_deposits'].clip(0, 100)

    # fill na here with 0, can't be NaN just indicates lack of data
    df['transaction_as_pct_of_balance'] = df['transaction_as_pct_of_balance'].fillna(0)
    df['max_withdrawals_30d'] = df['max_withdrawals_30d'].fillna(0)
    df['max_deposits_30d'] = df['max_deposits_30d'].fillna(0)
    df['pct_returned_deposits'] = df['pct_returned_deposits'].fillna(0)
    df['bal_ratio'] = df['bal_ratio'].fillna(0)
    df['sum_deposits_10d'] = df['sum_deposits_10d'].fillna(0)
    df['mean_account_balance_30d'] = df['mean_account_balance_30d'].fillna(0)
    df['dollar_val_dd'] = df['dollar_val_dd'].fillna(0) 
    df['nr_direct_deposits'] = df['nr_direct_deposits'].fillna(0)
    df['nr_past_transactions'] = df['nr_past_transactions'].fillna(0)
    df['dollar_val_returns'] = df['dollar_val_returns'].fillna(0)
    df['dollar_val_returns_3d'] = df['dollar_val_returns_3d'].fillna(0)
    df['nr_returns_30d'] = df['nr_returns_30d'].fillna(0)

    # fill na here with -1 indicating that this is the first ever transaction/giact never linked
    df['time_since_last_transaction'] = df['time_since_last_transaction'].fillna(-1)
    # df['giact_time_since_last_link'] = df['giact_time_since_last_link'].fillna(-1)
    df['giact_nr_decline'] = df['giact_nr_decline'].fillna(-1)
    df['giact_nr_other'] = df['giact_nr_other'].fillna(-1)
    df['nr_trans_ratio'] = df['nr_trans_ratio'].fillna(-1)
    df['first_deposit_amount'] = df['first_deposit_amount'].fillna(-1)
    df['lag_acc_open_first_transaction'] = df['lag_acc_open_first_transaction'].fillna(-1)
    
    df, _ = encode_defaults(df, default_values)
    
    df['all7120_default_encoded'] = df['all7120_default_encoded'].fillna(-1)
    df['bcc7120_default_encoded'] = df['bcc7120_default_encoded'].fillna(-1)
    
    return df


def get_test_df(scoring_data_base,
               scoring_data_files):
    """customize based on tasks
    
    Load and process test dataframe
    
    @params scoring_data_base: str
        - base path to data directory
    @params scoring_data_files: List(Dict)
        - list of dictionaries, each dict contains
            "fpath", "fname", etc
    """
    
    for sdfd in scoring_data_files:
        df = pd.read_parquet(os.path.join(scoring_data_base, sdfd["fpath"]))
        df = preprocess(df)
        print(f"{sdfd['fname']} preprocessed")

    active_accounts = df[
            (df.nr_past_transactions > 0) & (df.nr_transactions_30d > 0)
        ].user_id.unique()

    df.loc[:, "is_active"] = df.user_id.isin(active_accounts)

    # flip the sign to get positive corr with riskiness
    df.loc[:, "fico_adjusted"] = (df["fico_score"] 
                                  * np.where(df["fico_score"] > 850, 0, 1))
    df.loc[:, "fico_adjusted_pred"] = -df["fico_adjusted"]
#     df = df[df.is_active]

    return df


def train(modeling_df,
          date_col="sample_date", 
          indeterminate_col="indeterminate"):
    """
    Train and save model.
    """
    
    targ_cols = []
    for t in targets:
        targ_cols.extend(t.values())
    
    other_cols = []
    if indeterminate_col is not None:
        other_cols.append(indeterminate_col)
    if date_col is not None:
        other_cols.append(date_col)
        
    modeling_df = modeling_df[cols_raw+cols_raw_ni+targ_cols+other_cols]
    
    if 'indeterminate' in modeling_df.columns:
        print("data statistics", modeling_df.shape)
        print("indeterminate removed!")
        modeling_df = modeling_df[~modeling_df['indeterminate']]
        print("data statistics after removing indeterminate", modeling_df.shape)
        
    
    modeling_df = preprocess(modeling_df)
    print(f"modeling_df last date: {modeling_df[date_col].max()}")
    
    # get modeling data
    # data after Jan 16st 2019 when we started pulling GIACT
#     modeling_df = modeling_df[modeling_df['transaction_datetime'].between(pd.to_datetime('2019-01-16'),
#                                                                           pd.to_datetime('2019-12-31'))]

    # model hyper params
    seed = 15556
    print(seed)
    params = {
        "objective" : "binary",
        "metric" : "auc",
        "boosting": 'gbdt',
        "num_iterations": 1400,
        "max_depth" : 4,
        "num_leaves" : 15,
        "learning_rate" : 0.03,
        "feature_fraction" : 0.8,
        "subsample": 0.94,
        "lambda_l1": 16,
        "lambda_l2": 10,
        "min_data_in_leaf": 116,
        "tree_learner": "feature",
        "bagging_seed" : seed,
        "verbosity" : 1,
        "seed": seed,
        "categorical_feature": [39, 40, 41]  # with giact_time_since_last_link [40, 41, 42]
    }
    

    for target_definion in targets:
        target_col_name = target_definion['target_col_name']
        if 'ind_col_name' in target_definion:
            ind_col_name = target_definion['ind_col_name']
        else:
            ind_col_name = None
        
        if ind_col_name:
            mdf = modeling_df[~modeling_df[ind_col_name]]
        else:
            mdf = modeling_df.copy()
            
        print("target counts")
        print(mdf[target_col_name].value_counts())
        
        count_pos = mdf[target_col_name].sum()
        count_neg = (~mdf[target_col_name]).sum()
        pos_wgt_scaling_factor = count_neg / count_pos
        
        params['scale_pos_weight'] = pos_wgt_scaling_factor
        
        X = mdf[features]
        y = mdf[target_col_name]
    
        clf = lgb.LGBMClassifier(**params)
        clf = clf.fit(X, y)
        
    
    return clf

if __name__ == '__main__':
    train()
