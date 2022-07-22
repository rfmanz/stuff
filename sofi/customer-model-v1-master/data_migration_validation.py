# TODO:
# write something that automates data validation
# do in own workspace
# from cinfig file with queries pull data, then descirbe describe_df

# !sdm login

import json
import snowflake.connector
import pandas as pd

src_base = os.path.dirname(os.path.realpath(__file__))
# config_file = os.path.abspath(os.path.join(src_base, "../config.json"))
config_file_path_postgres = os.path.abspath(
    os.path.join(src_base, "../config_postgres.json")
)

# with open(config_file_path, "r") as f:
#     CONFIG_FILE = json.load(f)


with open(config_file_path_postgres, "r") as f:
    CONFIG_FILE_POSTGRES = json.load(f)


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


# def query_data_snowflake():
#     """
#     Get raw data with one or more queries.
#     """
#     dfs = {}
#     for name, qinf in CONFIG_FILE["sql_query_files"].items():
#         with open(os.path.join(src_base, qinf["query"])) as f:
#             query = f.read()

#         print("Running query {}.".format(name))
#         dfs[name] = run_query(query=query)

#     return dfs


def query_data_postgres():
    """
    Get raw data with one or more queries.
    """
    dfs = {}
    for name, qinf in CONFIG_FILE_POSTGRES["sql_query_files"].items():
        with open(os.path.join(src_base, qinf["query"])) as f:
            query = f.read()

        print("Running query {}.".format(name))
        pandas_df_to_s3(
            df=run_query(query=query), file_name=f"postgres_data/{name}.parquet"
        )


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


df =  pd.read_parquet(os.path.join('C:/Users/rfrancis/Downloads/','df_final2.parquet.gzip'))

list(df)
df.filter(regex="(?i)credit").columns
# df.shape


# for name, qinf in CONFIG_FILE["sql_query_files"].items():
#         with open(os.path.join(src_base, qinf["query"])) as f:
#             query = f.read()

# for name, qinf in CONFIG_FILE["sql_query_files"].items():


# CONFIG_FILE["sql_query_files"].keys()
# CONFIG_FILE.keys()
# list(CONFIG_FILE["sql_query_files"].items())[0]
# list(CONFIG_FILE["sql_query_files"].values())[0]
# list(CONFIG_FILE["sql_query_files"].keys())[0]
