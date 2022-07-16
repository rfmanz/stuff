import pandas as pd
import numpy as np
from time import time
import snowflake.connector
import sys
from typing import Optional, Tuple, NamedTuple, Union, Any, List, Dict, Type
from tabulate import tabulate    
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity= "all"
InteractiveShell.ast_node_interactivity = 'last_expr_or_assign'

# desired_width = 300
# pd.set_option('display.width', desired_width)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
import warnings
warnings.filterwarnings('ignore')

rmh_m = "tdm_risk_mgmt_hub.modeled."
rmh_c = "tdm_risk_mgmt_hub.cleansed."
tdm_bank_m = "tdm_bank.modeled."
tdm_bank_c = "tdm_bank.cleansed."
tdm_risk_c = 'TDM_RISK.CLEANSED.'
tdm_risk_m = 'TDM_RISK.modeled.'

s3_bucket = 's3://sofi-data-science/rarevalo/'

# pl_guardinex = pd.read_parquet('s3://sofi-data-science/rarevalo/guardinex_data_pull/pl_guardinex.parquet')

# t.filter(regex='(?i)banking')

# peek(run_query(check_table(tdm_bank_c,'profile_deposits')))

# time.asctime( time.localtime(1655996157) )

# find . -name "*1657*" -type d -exec rm -r {} \;

# chmod 0600 ~/.ssh/id_rsa
# eval `ssh-agent -s`
# ssh-add


def peek(df, rows=3):
    concat1 = pd.concat([df.dtypes, df.iloc[:3, :].T], axis=1).reset_index()
    concat1.columns = [''] * len(concat1.columns)
    return concat1

def check_table(data_source: object  = None, table_name : str = None )  -> str :
    return (f'select * from {data_source}{table_name} limit 5;')

def size_in_memory(df):
    return print(f"{(sys.getsizeof(df)/1024**2):.2f} Mb")


def get_snowflake_connection():
    ctx = snowflake.connector.connect(
        user="<username>",
        password="<password>",
        host="localhost",
        port=1444,
        account="sdm",
        warehouse="DATA_SCIENCE",
        database="DW_PRODUCTION",
        protocol="http",
    )
    return ctx


def run_query(sql):
    with get_snowflake_connection() as ctx:
        with ctx.cursor() as cs:
            cs.execute(sql)
            allthedata = cs.fetch_pandas_all()
            return allthedata


def view_tables(sql):
    with get_snowflake_connection() as ctx:
        with ctx.cursor() as cs:
            cs.execute(sql)
            allthedata = cs.fetchall()
            return [allthedata[i][1] for i in range(len(allthedata))]
           


def pandas_df_to_s3(
    df,
    bucket: str = "sofi-data-science",
    s3_path: str = "rarevalo",
    file_name: str = None,
    file_format: str = "parquet",
    **kwargs,
):
    """
    pandas_df_to_s3(df=df, file_name ='guardinex_data_pull/pl_guardinex.parquet')
    """
    import s3fs, boto3

    boto3.setup_default_session()
    s3 = s3fs.S3FileSystem(anon=False)

    if file_format in ["feather", "parquet"]:
        with s3.open(f"{bucket}/{os.path.join(s3_path,file_name)}", "wb") as f:
            getattr(df, f"to_{file_format}")(f, **kwargs)
    elif file_format in ["csv"]:
        with s3.open(f"{bucket}/{os.path.join(s3_path,file_name)}", "w") as f:
            getattr(df, f"to_{file_format}")(f, **kwargs)
    else:
        raise NotImplemented
        
        
def pd_options():
    desired_width = 300
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', None)
    import warnings
    warnings.filterwarnings('ignore')

def plot_single_numerical(df):
    plt.figure()
    sns.kdeplot(df, color="black", shade="gray")
    return plt.show()
    
    
def memory_usage_mb(df, *args, **kwargs):
    """Dataframe memory usage in MB."""
    return df.memory_usage(*args, **kwargs).sum()


def reduce_memory_usage(df, deep=True, verbose=True, categories=True):
    # All types that we want to change for "lighter" ones.
    # int8 and float16 are not include because we cannot reduce
    # those data types.
    # float32 is not include because float16 has too low precision.
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = memory_usage_mb(df, deep=deep)

    for col, col_type in df.dtypes.iteritems():
        best_type = None
        if col_type == "object":
            df[col] = df[col].astype("category")
            best_type = "category"
        elif col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name
        # Log the conversion performed.
        # if verbose and best_type is not None and best_type != str(col_type):
        # print(f"Column {col} converted from {col_type} to {best_type}")

    if verbose:
        end_mem = memory_usage_mb(df, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(
            f"Memory usage decreased from"
            f" {start_mem:.2f}MB to {end_mem:.2f}MB"
            f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)"
        )
