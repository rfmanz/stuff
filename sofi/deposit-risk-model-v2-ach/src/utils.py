import pandas as pd
import numpy as np
import functools
import json
from collections import defaultdict


def encode_special(
    df: pd.DataFrame, feature: str, interval: pd.Interval, encode_special: bool
):
    """
    Replace special values (beyond the provided interval inclusive) with NaN.
    If encode_special set to True, int encode them to another column.
    """
    # set up variables
    k = feature
    v = interval
    encode = encode_special
    df = df[feature].copy(deep=True).to_frame()
    cname = k + "_encoded"

    if isinstance(v, pd.Interval):
        is_default = ~df[k].between(v.left, v.right) & ~df[k].isna()
    elif isinstance(v, list):
        is_default = df[k].isin(k)
    else:
        raise RuntimeError("Data type {} not supported".format(str(type(v))))

    if ~is_default.isna().all():
        if encode:
            df.loc[is_default, cname] = is_default * df[k]
        df.loc[is_default, k] = np.nan  # set default values to NaN

    feature_col = df[feature]

    encoded_col = None
    if encode:
        encoded_col = df[cname]
    return feature_col, encoded_col


def get_timestamp_str():
    import time

    return str(int(time.time()))


def get_file_stdout_logger(log_dir, log_file, name=None, append_dir_timestamp=True):
    """
    Get default logger that both saves to file and log to stdout

    @params log_dir: directory to store the log
    @params log_file: name of the log
    @params name: parameter for logging.getLogger()
    @params append_dir_timestamp: append timestamp if True as ID

    @returns logger: the logger object
    """
    import os, sys, logging, datetime as dt

    # set dir
    if append_dir_timestamp:
        tstamp = int(dt.datetime.now().timestamp())
        log_dir = f"{log_dir}_{tstamp}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # build logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s: %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
    )

    # to avoid outputing multiple times to stdout
    if not logger.hasHandlers():
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def upload_s3(
    bucket_name,
    path_local,
    path_s3,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None,
    region_name=None,
):
    """Upload local file to S3.

    Parameters
    ----------
    bucket_name : string
        Bucket from which to download the file.
    path_local:
        Local path where to the file that will be uploaded.
    path_s3 :
        Path where the file will be uploaded in S3 bucket.

    Returns
    -------
    """
    import boto3

    boto3.setup_default_session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=None,
        region_name=None,
    )

    with open(path_local, "rb") as file:
        bucket = boto3.resource("s3").Bucket(bucket_name)
        bucket.put_object(Key=path_s3, Body=file)


import pandas as pd

from sqlalchemy import create_engine


def query_postgres(query, port, database="postgres"):
    """Query a postgres database using a string query and port.

    Parameters
    ----------
    query : string
        Query to be exectued.
    port : int/string
        Port through which the database is connected.

    Returns
    -------
    df : pandas DataFrame
    """
    engine = create_engine("postgres://127.0.0.1:{}/{}".format(str(port), database))
    _ = engine.connect()

    return pd.read_sql(query, engine)


def query_athena(query, port, database):
    raise NotImplementedError


def filter_by_prefix(L, prefix):
    return [l for l in L if l.startswith(prefix)]


def _get_config_file_field(config, field):
    """
    Get field value in config file.
    """
    return config[field]


def _set_config_file_field(config, field, val):
    """
    Set field value in config file.
    """
    config[field] = val

    with open(config_file_path, "w") as f:
        json.dump(config, f, indent=4)


##################################################
#              Saving and Loading
##################################################


def _save_dataframes(
    config, dataframes, base_path, prefix, timestamp_str=None, include_subdir=False
):
    import os, time

    if timestamp_str is None:
        timestamp_str = str(int(time.time()))

    if "data" not in config:  # init record keeping
        config["data"] = {}

    if prefix not in config["data"]:
        config["data"][prefix] = {}

    # prefix-path

    ppath = prefix
    if prefix:
        if len(dataframes) > 1 or include_subdir:
            ppath = os.path.join(prefix, f"{prefix}_{timestamp_str}")

    os.makedirs(os.path.join(base_path, ppath), exist_ok=True)

    for name, df in dataframes.items():

        fname = f"{name}_{timestamp_str}.parquet"
        fpath = os.path.join(base_path, ppath, fname)

        _to_parquet(df, fpath)
        config["data"][prefix][name] = os.path.join(ppath, fname)

    return config

def _load_dataframes(config, prefix, base_path, use_dask=False):
    import os
    from tqdm import tqdm

    if use_dask:
        import dask.dataframe as pd
    else:
        import pandas as pd

    dataframes = {}

    for name, path in tqdm(config["data"][prefix].items()):
        data_path = os.path.join(base_path, path)
        dataframes[name] = pd.read_parquet(data_path)
    return dataframes


# def _load_dataframes_dask(config, prefix, base_path):
#     import os, dask.dataframe as dd, pandas as pd
#     from dask import delayed
#
#     dataframes = {}
#     for name, path in config["data"][prefix].items():
#         data_path = os.path.join(base_path, path)
#         dataframes[name] = dd.read_parquet(data_path)
#     return dataframes


def _load_dataframe(config, prefix, name, base_path="data"):
    """
    Load individual dataframe from path in config file
    """
    import os, pandas as pd

    path = config["data"][prefix][name]
    return pd.read_parquet(os.path.join(base_path, path), use_threads=-1)


# not useful...just as a practise
# def load_save_dfs(func, *, params=defaultdict(lambda x: None)):
#     @functools.wraps(func)

#     config = params["config"]
#     prefix_in = params["prefix_in"]
#     prefix_out = params["prefix_out"]
#     base_path = params["base_path"]
#     debug = params["debug"]
#     timestamp_str = params["timestamp_str"]
#     include_subdir = params["include_subdir"]

#     if None in [config, prefix_in, prefix_out, base_path]:
#         raise ValueError("must provide config, prefix_in, prefix_out")

#     def wrapper_load_save_dfs(dfs, *args, **kwargs):
#         dfs = _load_dataframes(config, prefix_in, base_path)
#         dfs = func(dfs, *args, **kwargs)
#         config = _save_dataframes(config, dfs, base_path, prefix_out, timestamp_str, include_subdir)
#         return config

#     return wrapper_load_save_dfs


def _to_parquet(df: pd.DataFrame, path):
    """
    Handle bad formats.
    """
    import datetime

    for col in df.columns:
        val = df.head()[col].iloc[0]
        if isinstance(val, datetime.time):
            df[col] = df[col].astype(str)


    df = df.loc[:, ~df.columns.duplicated()]

    if "level_0" in df.columns:
        df = df.drop("level_0", axis=1)

    # df = df.reset_index(drop=False)
    df.to_parquet(path)


##################################################
#              MISC
##################################################


def add_prefix(df: pd.DataFrame, prefix, append_exist=False, exclusion=[]):
    """
    utility to append prefix to df columns

    Do not append if
    1. append_exist = True and the column starts the prefix already
    2. column is in exclusions (for e.g. don't append for ids/time index)

    Append if all other cases.
    """
    cols = df.columns
    result = []
    for c in cols:
        if (not append_exist and c.startswith(prefix)) or c in exclusion:
            col = c
        else:
            col = f"{prefix}{c}"
        result.append(col)
    df.columns = result
    return df


def get_data_dir(config, base_path, prefix_in):
    from pathlib import Path
    import os
    
    paths = config["data"][prefix_in]
    paths = list(set([str(Path(p).parents[0]) for p in paths.values()]))
    
    # should only have one directory
    assert(len(paths) == 1)
    path = os.path.join(base_path, paths[0])
    return path


# remove prefixs of columns
def remove_prefix(s: str, prefix: str) -> str:
    if s.startswith(prefix):
        return s[len(prefix):]
    else:
        return s[:]
    
def remove_prefixs(cols, prefixs=["ecp", "trnx", "socure", "banking", "tmx", "bk_acct_rstr"],
                   exclusions=["trnx_created_dt", "giact_created_dt", "socure_created_dt",
                               "trnx_is_return", "is_return"]):
    """
    remove prefixs from the list
    
    for this task, do not exclude plaid_ and giact_ by default
    b/c we had it included since the very beginning
    
    Dont judge...too lazy to optimize this function
    """
    cols_ = []
    prefixs = list(map(lambda x: x+"_", prefixs))
    for c in list(cols):
        
        if c in exclusions:
            cols_.append(c)
        
        # if c is not exluded, we need to find it
        found = c in exclusions
        for prefix in prefixs:
            if found:
                continue
            c_ = remove_prefix(c, prefix)
            if len(c_) < len(c):
                if "is_return" in c:
                    print(c, c_, prefix)
                cols_.append(c_)
                found = True
        if not found:
            cols_.append(c)
    
    assert(len(cols_) == len(cols))
    return cols_


# drop non ACH types
def drop_non_ach(df):
    df = df[df['trnx_transaction_code'].isin(['ACHDD']) & (df['trnx_transaction_amount'] > 0)]
    return df

def drop_non_check(df):
    df = df[df['trnx_transaction_code'].isin(['DDCK']) & (df['trnx_transaction_amount'] > 0)]
    return df


####################################
# Score scaling
####################################
    
def deposit_v1_scale_score(probability):
    """Returns a score based on a calculated formula for a probability.

    Args:
        probability (float): model probability returned from model predict_proba
    Returns:
        score (float): model score based on logit formula.
    Raises:
        AssertionError: if probability is not in the range [0, 1] (inclusive).

    """
    try:
        assert ((probability >= 0) & (probability <= 1)).all(), 'probability must be in range [0,1]'
    except AssertionError:
        raise

    # Formula parameters
    ALPHA = 50.92256438377247
    BETA = 2.9437347453224643

    # Minimum and maximum values for validation
    MINIMUM = 0.0
    MAXIMUM = 100

    # Convert probability to score.
    score = np.minimum(np.maximum(np.log(probability / (1 - probability)) * BETA + ALPHA, MINIMUM), MAXIMUM)

    return score 

def scale_ach_v2_dev(probability):
    """Returns a score based on a calculated formula for a probability.

    Args:
        probability (float): model probability returned from model predict_proba
    Returns:
        score (float): model score based on logit formula.
    Raises:
        AssertionError: if probability is not in the range [0, 1] (inclusive).

    """
    try:
        assert ((probability >= 0) & (probability <= 1)).all(), 'probability must be in range [0,1]'
    except AssertionError:
        raise

    # Formula parameters    
    ALPHA = 49.80051053449891
    BETA = 7.393401999397325

    # Minimum and maximum values for validation
    MINIMUM = 0.0
    MAXIMUM = 100

    # Convert probability to score.
    score = np.minimum(np.maximum(np.log(probability / (1 - probability)) * BETA + ALPHA, MINIMUM), MAXIMUM)

    return score 

def scale_ach_v2_refit(probability):
    """Returns a score based on a calculated formula for a probability.

    Args:
        probability (float): model probability returned from model predict_proba
    Returns:
        score (float): model score based on logit formula.
    Raises:
        AssertionError: if probability is not in the range [0, 1] (inclusive).

    """
    try:
        assert ((probability >= 0) & (probability <= 1)).all(), 'probability must be in range [0,1]'
    except AssertionError:
        raise

    # Formula parameters    
    ALPHA = 49.56347264658153
    BETA = 7.520032961169707

    # Minimum and maximum values for validation
    MINIMUM = 0.0
    MAXIMUM = 100

    # Convert probability to score.
    score = np.minimum(np.maximum(np.log(probability / (1 - probability)) * BETA + ALPHA, MINIMUM), MAXIMUM)

    return score 

def build_score_coefficients(pred):
    """
    For converting probability to score ranging from 0 to 100 using this formula:

    scores = log(preds / (1 - preds)) * a + b

    Where a and b are:
    a = 100 / (max - min)
    b = - (100 * min) / max

    Where max and min are: 
    max = max(log(preds / (1 - preds)))
    min = min(log(preds / (1 - preds)))
    
    
    call a, b = build_score_coefficients(pred) 
    """
    scores = np.log(pred / (1 - pred))
    s_max = scores.max()
    s_min = scores.min()
    a = 100 / (s_max - s_min)
    b = (100 * s_min) / (s_min - s_max)
    return a, b

