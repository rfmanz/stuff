# make this a meta class for all things data. so loader & saver

import pandas as pd
import numpy as np
from time import time
import snowflake.connector
import sys
from attrs import asdict, define, make_class, Factory
import functools
from typing import Optional, Tuple, NamedTuple, Union, Any, List, Dict, Type
from tabulate import tabulate
from sagemaker import get_execution_role
import s3fs, boto3


# @define
class dataloader:

    """Data loader"""

    # path: None
    def __init__(
        self,
        config_file_path: str = ("config.json",),
        debug: bool = (False,),
        sql: str = None,
    ):

        self.sql = sql
        self.allthedata = None
        self.categorical = None
        self.numerical = None

        # self.df = None

        # if config_file_path:
        #   with open(config_file_path, "r") as f:
        #     CONFIG_FILE = json.load(f)

    def get_snowflake_connection(self):

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

    def run_query(self, query=None):
        if self.sql is None:
            self.sql = query
        with self.get_snowflake_connection() as ctx:
            with ctx.cursor() as cs:
                cs.execute(self.sql)
                self.allthedata = cs.fetch_pandas_all()
                return self.allthedata

    def describe_df(self, query=None, floatfmt=".3f"):
        # Numerical
        df = self.run_query(query)
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
        concatenated_numerical = concatenated_numerical.iloc[
            :, [0, 4, 5, 6, 7, 1, 2, 3]
        ]

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
                #                    colalign=("right"),
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
                        round(categorical.isnull().sum() / df.shape[0] * 100, 2).astype(
                            str
                        )
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
            self.categorical = concatenated_categorical
            self.numerical = concatenated_numerical
            return (
                # print("concatenated_numerical"),
                # print("concatenated_categorical"),
                concatenated_numerical,
                concatenated_categorical,
            )


# def _load(self, df, fname, suffix="parquet", **kwargs):
#     """
#     df (pd.DataFrame): dataframe to save
#     fname (str): saved file will take form "{self.dir_path}/{fname}{suffix}"
#     suffix (str): saved file will take form "{self.dir_path}/{fname}{suffix}"
#     kwargs: to be passed into df.to_parquet(**kwargs)
#     """

#     fpath = os.path.join(self.dir_path, f"{fname}.{suffix}")
#     df.to_parquet(fpath, **kwargs)


def pandas_df_to_s3(
    df,
    bucket: str = "sofi-data-science",
    s3_path: str = 'rarevalo',
    file_name: str = None,
    file_format: str = "parquet",
    **kwargs,
):
    """
    Upload Pandas DataFrame to S3

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to upload
    bucket: string
        Bucket from which to upload the file: e.g. sofi-data-science
    s3_path: string
        Your S3 folder 
    file_name: string
        Name of the file        
    file_format: string
        Format to store the df in, currently supports feather, parquet, and csv.
    Returns
    -------
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


def s3_to_pandas_df(file_name):

    role = get_execution_role()
    bucket = "sofi-data-science"
    key = "rarevalo/"
    data_location = f"s3://{bucket}/{key}"
    df = pd.read_parquet(data_location)
    return df


# class DataDumper:
#     def __init__(self, dir_path):
#         """
#         dir_path (str): path to directory
#         """
#         self.dir_path = dir_path
#         os.makedirs(dir_path, exist_ok=True)

#     def to_parquet(self, df, fname, suffix="parquet", **kwargs):
#         """
#         df (pd.DataFrame): dataframe to save
#         fname (str): saved file will take form "{self.dir_path}/{fname}{suffix}"
#         suffix (str): saved file will take form "{self.dir_path}/{fname}{suffix}"
#         kwargs: to be passed into df.to_parquet(**kwargs)
#         """
#         fpath = os.path.join(self.dir_path, f"{fname}.{suffix}")
#         df.to_parquet(fpath, **kwargs)

#     def to_parquets(
#         self, dfs, fname, append_timestamp=True, suffix="parquet", **kwargs
#     ):
#         if not append_timestamp:
#             raise NotImplementedError("Currently only supports append_timestamp=True")

#         for df_ in tqdm.tqdm(dfs):
#             dt_str = str(dt.datetime.now().timestamp())
#             fname_ = f"{fname}_{dt_str}"
#             self.to_parquet(df_, fname_, suffix=suffix, **kwargs)

#     @staticmethod
#     def get_timestamp_str(round=True):
#         if round:
#             return str(int(dt.datetime.now().timestamp()))
#         return str(dt.datetime.now().timestamp())
