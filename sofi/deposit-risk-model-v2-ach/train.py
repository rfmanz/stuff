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

MODELS_BASE_PATH = '/home/ec2-user/SageMaker/money-risk-models/money-transactional-data/models'
MODELING_DATA_PATH = '/home/ec2-user/SageMaker/money-risk-models/money-transactional-data/data/labeled/labeled_1574464092.feather'

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


# target col name
target_col_name = 'target_1'

# indeterminate col name
ind_col_name = 'indeterminate_1'

# Model lt 10d.
cols_raw_lt_10d = ['vantage_score', 'fraud_score_1', 'email_risk_score', 'credit_card_loan_amount', 
                   'fraud_score_2', 'phone_risk_score', 'all8220', 'lag_acc_open_first_transaction', 
                   'all7120', 'bcc7120', 'total_outstanding_balance', 'name_address_correlation', 
                   'address_risk_score', 'name_email_correlation', 'total_tradelines_open', 
                   'iln5520', 'iqt9415', 'giact_nr_decline', 'education_loan_amount', 'nr_direct_deposits',
                   'transaction_amount', 'sum_deposits_10d', 'time_since_last_transaction', 'nr_social_profiles_found',
                   'giact_nr_pass', 'giact_time_since_last_link', 'giact_time_since_first_link', 
                   'name_phone_correlation']

cols_raw_ni_lt_10d = ['transaction_code', 'transaction_datetime', 'days_since_first_deposit']

cols_encoded_lt_10d = ['transaction_code_encoded', 'all7120_default_encoded', 'bcc7120_default_encoded']

features_lt_10d = cols_raw_lt_10d + cols_encoded_lt_10d

default_values_lt_10d = {
    'vantage_score': [pd.Interval(300, 850), False],
    'all7120': [pd.Interval(0, 990), True],
    'all8220': [pd.Interval(0, 9990), False],
    'bcc7120': [pd.Interval(0, 990), True],
    'iln5520': [pd.Interval(0, 999999990), False],
}

def preprocess_lt_10d(df):
    """
    Code to preprocess lt_10d model.
    """
    # mapping from transaction code to integer
    mdict = {'ACHDD': 0, 'ACHDDIN': 1, 'DD': 2, 'DDCK': 3, 'ACHINDD': 4}
    df['transaction_code_encoded'] = df['transaction_code'].map(mdict)
    
    # fill na here with -1 indicating that this is the first ever transaction/giact never linked
    df['time_since_last_transaction'] = df['time_since_last_transaction'].fillna(-1)
    df['giact_time_since_last_link'] = df['giact_time_since_last_link'].fillna(-1)
    df['giact_time_since_first_link'] = df['giact_time_since_first_link'].fillna(-1)
    df['giact_nr_decline'] = df['giact_nr_decline'].fillna(-1)
    df['giact_nr_pass'] = df['giact_nr_pass'].fillna(-1)

    # ignored 'DD' transaction codes in feature engineering, fixing here (temp)
    df[['sum_deposits_10d']].fillna(df['transaction_amount'], inplace=True)
    
    df, _ = encode_defaults(df, default_values_lt_10d)
    
    return df


def train_lt_10d():
    """
    Train and save lt_10 model.
    """
    modeling_df = pd.read_feather(MODELING_DATA_PATH, 
                                  columns=cols_raw_lt_10d+cols_raw_ni_lt_10d+[target_col_name]+[ind_col_name], 
                                  use_threads=-1)
    modeling_df = preprocess_lt_10d(modeling_df)
    
    monotone_constraints = [-1, 0, 1, 0, 1, 1] + [0 for i in range(len(features_lt_10d) - 6)]
    
    # get modeling data
    # data after Jan 16st 2019 when we started pulling GIACT
    modeling_df = modeling_df[~modeling_df[ind_col_name] & (modeling_df['transaction_datetime'] >= pd.to_datetime('2019-01-16'))]

    # data with less than 10 days since first deposit
    modeling_df = modeling_df[modeling_df['days_since_first_deposit'] < 10]

    count_pos = modeling_df[target_col_name].sum()
    count_neg = (~modeling_df[target_col_name]).sum()
    pos_wgt_scaling_factor = count_neg / count_pos
    
    seed = 15556
    
    params = {
        "objective" : "binary",
        "metric" : "auc",
        "boosting": 'gbdt',
        "num_iterations": 3400,
        "max_depth" : 4,
        "num_leaves" : 15,
        "learning_rate" : 0.04,
        "feature_fraction" : 0.85,
        "subsample": 0.95,
        "lambda_l1": 10,
        "lambda_l2": 2,
        "min_data_in_leaf": 120,
        "scale_pos_weight": pos_wgt_scaling_factor,
        "tree_learner": "feature",
        "boost_from_average": "false",
        "bagging_seed" : seed,
        "verbosity" : 1,
        "seed": seed,
        "monotone_constraints": monotone_constraints
    }

    X = modeling_df[features_lt_10d]
    y = modeling_df[target_col_name]
    
    clf = lgb.LGBMClassifier(**params)
    clf = clf.fit(X, y)
        
    dir_name = 'model_lt_10d'
    model_name = 'deposit_risk_lt_10d_' + get_timestamp_str() + '.pkl'
    pickleobj(clf, os.path.join(MODELS_BASE_PATH, dir_name, model_name))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI for training the deposit risk model.',
                                     prefix_chars='-')
    parser.add_argument('-t', '--lt10d', action='store_true', default=False,
                        help='Train lt10d model.', dest='lt10d')
    args = parser.parse_args()
    
    if args.lt10d:
        train_lt_10d()
