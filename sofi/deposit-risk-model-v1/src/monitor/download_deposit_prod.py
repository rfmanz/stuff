# pip install --index-url https://build.sofi.com/artifactory/api/pypi/pypi/simple sofiproto==release-1703
# pip install read_protobuf
import pickle, sys, os, boto3, botocore
import numpy as np
import pandas as pd
import mdsutils

sys.path.append('/home/ec2-user/SageMaker/machine-learning/ml-deploys/customer-risk-v2/web/')

from sofiproto.moneyfraud import deposit_risk_v3_pb2
from read_protobuf import read_protobuf
from multiprocessing import Pool, cpu_count
from collections import defaultdict

s3 = boto3.client('s3')


def download_user_input_deposit(req_path):
    """
    given request path, go download input features to each proxy_id
    """
    if not isinstance(req_path, str): return 

    try:
        prbf = s3.get_object(Bucket='machine-learning-proxy.sofi.com', Key=req_path)["Body"].read()
        df_row = read_protobuf(prbf, deposit_risk_v3_pb2.DepositRiskInputV3())
        row_dict = df_row.to_dict('records')[0]
        row_dict["proxy_id"] = req_path.split("/")[1]
        return row_dict

    except botocore.exceptions.ClientError:
        pass

def download_deposit_v1_prod_data(output_path=None, debug=False):
    # previous version
#     money_user_query = """
#     select money_rule_result.money_rule_result_id,
#         money_rule_result.money_trigger_source_id,
#         money_rule_result.party_id,
#         money_rule_result.created_dt,
#         json_extract(money_cache_snapshot, '$.accountData["accountNumber"]') as accountNumber,
#         json_extract(rule_result_info, '$.rules["deposit risk rule"].score') as model_score,
#         json_extract(rule_result_info, '$.rules["deposit risk rule"].machineLearningProxyId') as proxy_id,
#         json_extract(rule_result_info, '$.rules["deposit risk rule"].result') as rule_result
#     from money_rule_result
#          left join money_trigger_source mts
#                    on money_rule_result.money_trigger_source_id = mts.money_trigger_source_id
#          left join money_action ma on money_rule_result.money_rule_result_id = ma.money_rule_result_id
#          left join money_cache_snapshot mcs
#                    on money_rule_result.money_cache_snapshot_id = mcs.money_cache_snapshot_id
#     where json_extract(rule_result_info, '$.rules["deposit risk rule"].result') is not null
#     and money_rule_result.created_dt >= TIMESTAMP '2019-10-19'; 
#     """ #  -- was launched on that date
    print("getting new version of data - query update date 1/19/2022")
    money_user_query = """
        with tbl as (
        select money_rule_result.money_rule_result_id,
               money_rule_result.money_trigger_source_id,
               money_rule_result.party_id,
               money_rule_result.created_dt,
               json_extract(money_cache_snapshot, '$.accountData["accountNumber"]') as accountNumber,
               case when json_extract(rule_result_info, '$.rules["deposit risk rule"].result') is not null
                    then json_extract(rule_result_info, '$.rules["deposit risk rule"]')
                    when json_extract(rule_result_info, '$.rules["DEPOSIT_RISK_RULE"].result') is not null
                    then json_extract(rule_result_info, '$.rules["DEPOSIT_RISK_RULE"]')
               end as deposit_risk_rule
         from money_rule_result
         left join money_trigger_source mts
                   on money_rule_result.money_trigger_source_id = mts.money_trigger_source_id
         left join money_action ma on money_rule_result.money_rule_result_id = ma.money_rule_result_id
         left join money_cache_snapshot mcs
                   on money_rule_result.money_cache_snapshot_id = mcs.money_cache_snapshot_id
         where (json_extract(rule_result_info, '$.rules["deposit risk rule"].result') is not null
               or json_extract(rule_result_info, '$.rules["DEPOSIT_RISK_RULE"].result') is not null)
         and money_rule_result.created_dt >= TIMESTAMP '2019-10-19'
    )
    select
           money_rule_result_id,
           money_trigger_source_id,
           party_id,
           created_dt,
           accountNumber,
           json_extract(deposit_risk_rule, '$.score') as model_score,
           json_extract(deposit_risk_rule, '$.machineLearningProxyId') as proxy_id,
           json_extract(deposit_risk_rule, '$.result') as rule_result
    from tbl
    """
    
    athena = mdsutils.AthenaClient(database='datalake_production_money_monitoring')
    print("querying money user data...")
    money_users_df = athena.query_to_df(money_user_query)
    
    if debug:
        money_users_df = money_users_df.head(5000)
    
    for col in ["proxy_id", "rule_result"]:
        money_users_df[col] = money_users_df[col].str.strip('"')
    
    money_users_df = money_users_df[money_users_df.rule_result != "ERROR"]
    
    # basically trying to not hard code "money-deposit-risk-v3"...
    money_users_df["model_name"] = money_users_df["proxy_id"].apply(lambda x: x.split("/")[0])
    money_users_df["proxy_id"] = money_users_df["proxy_id"].apply(lambda x: x.split("/")[1])
    money_users_df["req_path"] = money_users_df["model_name"] + "/" + money_users_df["proxy_id"] + "/request"

    # get inputs 
    print("querying input features by proxy_id")
    num_cores = cpu_count()
    with Pool(cpu_count()) as p:
        ret_list = p.map(download_user_input_deposit, 
                         money_users_df.req_path.tolist())

    ret_list = list(filter(lambda x: x is not None, ret_list))
    ret_df = pd.DataFrame(ret_list)
    ret_df = ret_df[~ret_df.proxy_id.isna()]
    full_df = money_users_df.merge(ret_df, how = 'left', on = 'proxy_id')
    
    if output_path:
        full_df.to_parquet(output_path)
    
    return full_df, money_users_df, ret_df