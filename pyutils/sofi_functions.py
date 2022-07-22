import pandas as pd
import os
import numpy as np
import time
import snowflake.connector
import sys
from typing import Optional, Tuple, NamedTuple, Union, Any, List, Dict, Type
from tabulate import tabulate
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
InteractiveShell.ast_node_interactivity = "last_expr_or_assign"

# desired_width = 300
# pd.set_option('display.width', desired_width)
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 500)
import warnings

warnings.filterwarnings("ignore")

rmh_m = "tdm_risk_mgmt_hub.modeled"
rmh_c = "tdm_risk_mgmt_hub.cleansed"
tdm_bank_m = "tdm_bank.modeled"
tdm_bank_c = "tdm_bank.cleansed"
tdm_risk_c = "TDM_RISK.CLEANSED"
tdm_risk_m = "TDM_RISK.modeled"

s3_bucket = "s3://sofi-data-science/rarevalo/"

# pl_guardinex = pd.read_parquet(os.path.join(s3_bucket,'guardinex_data_pull/pl_guardinex.parquet'))

# t.filter(regex='(?i)banking')

# time.asctime( time.localtime(1633038321) )

# find . -name "*1657*" -type d -exec rm -r {} \;

# chmod 0600 ~/.ssh/id_rsa
# eval `ssh-agent -s`
# ssh-add

# git commit -a -m '.'

# import sys
# sys.path.insert(1, '/home/ec2-user/SageMaker/Rafael/pyutils/pyutils/')
# from sofi_functions import *

# Reload kernal
# import os
# os._exit(00)

# import importlib
# importlib.reload(nameOfModule)

df = pd.read_parquet(
    os.path.join("C:/Users/rfrancis/Downloads/", "df_final2.parquet.gzip")
)


def peek(df, rows=3):
    concat1 = pd.concat([df.dtypes, df.iloc[:3, :].T], axis=1).reset_index()
    concat1.columns = [""] * len(concat1.columns)
    return concat1


def check_table(
    data_source: object = None,
    table_name: str = None,
    cols: List[Optional[str]] = None,
    t=True,
    print_sql=False,
) -> str:
    """'select {cols} from {data_source}.{table_name} limit 5"""
    if cols:
        if isinstance(cols, list):
            cols = ",".join(cols).replace(",", ",\n")
        _tbl = f"select {cols} from {data_source}.{table_name} limit 5;"
    else:
        _tbl = f"select * from {data_source}.{table_name} limit 5;"

    return print(_tbl) if print_sql else peek(run_query(_tbl)) if t else run_query(_tbl)


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


def show_tables(dw=None, like_tbl_name: str = None):
    with get_snowflake_connection() as ctx:
        with ctx.cursor() as cs:
            if like_tbl_name is None:
                cs.execute(f"show tables in {dw}")
            else:
                cs.execute(f"show tables like '{like_tbl_name}' in {dw}")
            allthedata = cs.fetchall()
            return f"{dw}", [allthedata[i][1] for i in range(len(allthedata))]


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
    pd.set_option("display.width", desired_width)
    pd.set_option("display.max_columns", None)
    import warnings

    warnings.filterwarnings("ignore")


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


def describe_df(df, return_df=None, floatfmt=".3f"):
    # Numerical
    print("--" * 20)
    print("Columns:", df.shape[1])
    print("Rows:", df.shape[0])
    print("Memory usage:", (f"({(sys.getsizeof(df) / 1024 ** 2):.2f} Mb)"))

    print("--" * 20)
    print("NUMERICAL VARIABLES:")

    numerical = df.select_dtypes(include=np.number)
    concatenated_numerical = (
        pd.concat(
            [
                round(numerical.isnull().sum() / df.shape[0] * 100, 2).astype(str)
                + "%",
                numerical.isnull().sum(),
                numerical.count(),
                numerical.min(),
                numerical.mean(),
                numerical.max(),
            ],
            axis=1,
            keys=["%NULLS", "COUNT_NULLS", "NOT_NULL", "MIN", "MEAN", "MAX"],
            sort=False,
        )
        .sort_values("COUNT_NULLS", ascending=False)
        .reset_index()
        .rename(columns={"index": ""})
    )

    t = numerical.mode().T
    t.rename(columns={0: "MODE"}, inplace=True)
    concatenated_numerical = concatenated_numerical.merge(
        t, how="left", left_on="", right_on=t.index
    )
    concatenated_numerical.index = concatenated_numerical.index + 1
    concatenated_numerical = concatenated_numerical.iloc[:, [0, 4, 5, 6, 7, 1, 2, 3]]

    print(
        tabulate(
            concatenated_numerical,
            headers=[
                "MIN",
                "MEAN",
                "MAX",
                "MODE",
                "%NULLS",
                "#_NULLS",
                "NOT_NULL",
            ],
            tablefmt="presto",
            colalign=("right"),
            floatfmt=floatfmt,
        )
    )

    # Categorical

    print("-----" * 20)
    print()
    print("CATEGORICAL VARIABLES:")
    categorical = df.select_dtypes(["object", "category"])
    if categorical.shape[1] == 0:
        print("No Categorical Variables")
    else:
        concatenated_categorical = (
            pd.concat(
                [
                    round(categorical.isnull().sum() / df.shape[0] * 100, 2).astype(str)
                    + "%",
                    categorical.isnull().sum(),
                    categorical.count(),
                ],
                keys=["%NULLS", "COUNT_NULLS", "NOT_NULL"],
                axis=1,
                sort=False,
            )
            .sort_values("%NULLS", ascending=False)
            .reset_index()
            .rename(columns={"index": ""})
        )

        max_unique = 5
        u_strs = []

        for col in categorical:
            series = categorical.loc[categorical[col].notnull(), col]
            n_unique = series.nunique()
            if n_unique > max_unique:
                u_strs.append(str(n_unique) + " unique values")
            else:
                u_strs.append(str(series.unique()))

        t = pd.DataFrame(u_strs, categorical.columns)
        t = t.reset_index()
        t = t.rename(columns={"index": "", 0: "Unique_Values"})
        concatenated_categorical = concatenated_categorical.merge(t, on="")
        concatenated_categorical.index = concatenated_categorical.index + 1

        print(
            tabulate(
                concatenated_categorical,
                headers=["%NULLS", "#_NULLS", "NOT_NULL", "Unique_Values"],
                tablefmt="presto",
                colalign=("left"),
            )
        )

    return (concatenated_numerical, concatenated_categorical) if return_df else None
