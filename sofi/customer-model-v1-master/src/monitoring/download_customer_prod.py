# pip install --index-url https://build.sofi.com/artifactory/api/pypi/pypi/simple sofiproto==release-1703
# pip install read_protobuf
import pickle, sys, os, boto3, botocore
import numpy as np
import pandas as pd
import mdsutils

sys.path.append('/home/ec2-user/SageMaker/machine-learning/ml-deploys/customer-risk-v2/web/')

from sofiproto.moneyfraud import customer_risk_v2_pb2
from read_protobuf import read_protobuf
from multiprocessing import Pool, cpu_count
from collections import defaultdict

s3 = boto3.client('s3')

cols_raw = ['first_deposit_amount', 'vantage_score', 'bcc7120', 'email_risk_score',
           'fraud_score_2', 'name_email_correlation', 'transaction_as_pct_of_balance',
           'mean_account_balance_30d', 'giact_time_since_last_link', 'phone_risk_score',
           'name_address_correlation', 'all8220', 'lag_acc_open_first_transaction',
           'dollar_val_dd', 'all7120', 'sum_deposits_10d', 'nr_past_transactions',
           'total_tradelines_open', 'education_loan_amount', 'address_risk_score',
           'iqt9415', 'max_withdrawals_30d', 'iln5520', 'max_deposits_30d',
           'pct_returned_deposits', 'giact_nr_decline', 'nr_direct_deposits',
           'time_since_last_transaction', 'bal_ratio', 'name_phone_correlation',
           'giact_nr_other', 'dollar_val_returns', 'nr_trans_ratio', 'iqt9413',
           'dollar_val_returns_3d', 'nr_returns_30d', 'credit_card_loan_amount',
           'fraud_score_1', 'age_money_account']

cols_raw_ni = ['transaction_code', 'transaction_datetime', 'days_since_first_deposit']

def preprocess(df):
    """
    Code to preprocess model.
    """
    for col in cols_raw:
        if col not in df.columns:
            df[col] = np.nan
    
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
    df['giact_time_since_last_link'] = df['giact_time_since_last_link'].fillna(-1)
    df['giact_nr_decline'] = df['giact_nr_decline'].fillna(-1)
    df['giact_nr_other'] = df['giact_nr_other'].fillna(-1)
    df['nr_trans_ratio'] = df['nr_trans_ratio'].fillna(-1)
    df['first_deposit_amount'] = df['first_deposit_amount'].fillna(-1)
    df['lag_acc_open_first_transaction'] = df['lag_acc_open_first_transaction'].fillna(-1)
    
    df, _ = encode_defaults(df, default_values)
    
    df['all7120_default_encoded'] = df['all7120_default_encoded'].fillna(-1)
    df['bcc7120_default_encoded'] = df['bcc7120_default_encoded'].fillna(-1)
    
    return df

default_values = {
    'vantage_score': [pd.Interval(300, 850), False],
    'all7120': [pd.Interval(0, 990), True],
    'all8220': [pd.Interval(0, 9990), False],
    'bcc7120': [pd.Interval(0, 990), True],
    'iln5520': [pd.Interval(0, 999999990), False],
    'iqt9413': [pd.Interval(0, 90), False]
}

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

def download_user_input(proxy_id):
    if not isinstance(proxy_id, str): return 
    req_path = 'data-science-engineering-machine-learning-customer-risk-v2/' + proxy_id + '/request'

    try:
        prbf = s3.get_object(Bucket='machine-learning-proxy.sofi.com', Key=req_path)["Body"].read()
        df_row = read_protobuf(prbf, customer_risk_v2_pb2.CustomerRiskInputV2())
        row_dict = df_row.to_dict('records')[0]
        row_dict["proxy_id"] = proxy_id
        return row_dict

    except botocore.exceptions.ClientError:
        pass
        
def download_customer_prod_data(output_path, debug=False): 
    
    # last record
    money_user_query = f"""
    with df_tmp as
        (select party_id, created_dt, updated_dt, last_checked_date,
            max(last_checked_date) over (partition by party_id) last_checked_date_max,
            json_extract(risk_group_decision_info, '$.newRiskGroup') as risk_group,
            json_extract(json_extract(risk_group_decision_info, '$.decisionContext'), '$.requestSource') as request_source,
            json_extract(json_extract(json_extract(risk_group_decision_info, '$.decisionContext'), '$.customerModelResult'), '$.modelScore') as model_score,
            json_extract(json_extract(json_extract(risk_group_decision_info, '$.decisionContext'), '$.customerModelResult'), '$.decisionRuleDetails') as model_decision_rule_details,
            json_extract(json_extract(json_extract(risk_group_decision_info, '$.decisionContext'), '$.customerModelResult'), '$.customerModelRiskGroup') as mcustomer_model_risk_groupodel_score,
            json_extract(json_extract(json_extract(risk_group_decision_info, '$.decisionContext'), '$.customerModelResult'), '$.machineLearningProxyId') as proxy_id,
            json_extract(json_extract(risk_group_decision_info, '$.decisionContext'), '$.riskGroupEvaluationMethod') as risk_group_evaluation_method,
            COALESCE (CAST(json_extract(json_extract(risk_group_decision_info, '$.decisionContext'), '$.partnerName') as VARCHAR), 'SOFI') as partner_name
        from datalake_production_money_users.risk_group_history)  -- if wanna debug, set limit here
    select * from df_tmp
    where last_checked_date_max = last_checked_date;
    """
        
    athena = mdsutils.AthenaClient(database='datalake_production_money_users')
    print("querying money user data...")
    money_users_df = athena.query_to_df(money_user_query)
    
    if debug:
        print("debugging mode on")
        money_users_df = money_users_df.head(1000)
    
    print("some data processing...")
    money_users_df.dropna(axis = 0, subset = ['proxy_id'], inplace = True)
    money_users_df['proxy_id'] = money_users_df['proxy_id'].apply(lambda x: x[59:].strip('/').strip('"'))
    money_users_df.dropna(axis = 0, subset = ['proxy_id'], inplace = True)

    # filter by date? If we are only monitoring new account, need to filter by created_dt
#     money_users_df.created_dt = pd.to_datetime(money_users_df.created_dt)
#     money_users_df = money_users_df[money_users_df.created_dt > pd.to_datetime(last_monitoring_date)]

    num_cores = cpu_count()
    with Pool(cpu_count()) as p:
        ret_list = p.map(download_user_input, 
                         [row["proxy_id"] 
                          for _, row in money_users_df.iterrows()])

    # this was breaking the pipeline
    ret_list = list(filter(lambda x: x is not None, ret_list))
    ret_df = pd.DataFrame(ret_list)
    ret_df = ret_df[~ret_df.proxy_id.isna()]
    full_df = money_users_df.merge(ret_df, how = 'left', on = 'proxy_id')
    full_df = preprocess(full_df)
    
    full_df.to_parquet(output_path)
    

def download_customer_prod_databy_id(output_path, uid, debug=False): 
    
    # last record
    money_user_query = f"""
    with df_tmp as
        (select party_id, created_dt, updated_dt, last_checked_date,
            max(last_checked_date) over (partition by party_id) last_checked_date_max,
            json_extract(risk_group_decision_info, '$.newRiskGroup') as risk_group,
            json_extract(json_extract(risk_group_decision_info, '$.decisionContext'), '$.requestSource') as request_source,
            json_extract(json_extract(json_extract(risk_group_decision_info, '$.decisionContext'), '$.customerModelResult'), '$.modelScore') as model_score,
            json_extract(json_extract(json_extract(risk_group_decision_info, '$.decisionContext'), '$.customerModelResult'), '$.decisionRuleDetails') as model_decision_rule_details,
            json_extract(json_extract(json_extract(risk_group_decision_info, '$.decisionContext'), '$.customerModelResult'), '$.customerModelRiskGroup') as mcustomer_model_risk_groupodel_score,
            json_extract(json_extract(json_extract(risk_group_decision_info, '$.decisionContext'), '$.customerModelResult'), '$.machineLearningProxyId') as proxy_id,
            json_extract(json_extract(risk_group_decision_info, '$.decisionContext'), '$.riskGroupEvaluationMethod') as risk_group_evaluation_method,
            COALESCE (CAST(json_extract(json_extract(risk_group_decision_info, '$.decisionContext'), '$.partnerName') as VARCHAR), 'SOFI') as partner_name
        from datalake_production_money_users.risk_group_history)  -- if wanna debug, set limit here
    select * from df_tmp
    where party_id={uid};
    """
        
    athena = mdsutils.AthenaClient(database='datalake_production_money_users')
    print("querying money user data...")
    money_users_df = athena.query_to_df(money_user_query)
    
    if debug:
        print("debugging mode on")
        money_users_df = money_users_df.head(1000)
    
    print("some data processing...")
    money_users_df.dropna(axis = 0, subset = ['proxy_id'], inplace = True)
    money_users_df['proxy_id'] = money_users_df['proxy_id'].apply(lambda x: x[59:].strip('/').strip('"'))
    money_users_df.dropna(axis = 0, subset = ['proxy_id'], inplace = True)

    # filter by date? If we are only monitoring new account, need to filter by created_dt
#     money_users_df.created_dt = pd.to_datetime(money_users_df.created_dt)
#     money_users_df = money_users_df[money_users_df.created_dt > pd.to_datetime(last_monitoring_date)]

    num_cores = cpu_count()
    with Pool(cpu_count()) as p:
        ret_list = p.map(download_user_input, 
                         [row["proxy_id"] 
                          for _, row in money_users_df.iterrows()])

    # this was breaking the pipeline
    ret_list = list(filter(lambda x: x is not None, ret_list))
    ret_df = pd.DataFrame(ret_list)
    ret_df = ret_df[~ret_df.proxy_id.isna()]
    full_df = money_users_df.merge(ret_df, how = 'left', on = 'proxy_id')
    full_df = preprocess(full_df)
    
    full_df.to_parquet(output_path)