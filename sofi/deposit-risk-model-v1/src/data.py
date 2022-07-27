"""
Obtain and clean data.
"""

import features

import argparse
import datetime
import gc
import os
import time
import json

import pandas as pd

from pandas.tseries.offsets import BDay
from rdsutils.aws import upload_s3

# from rdsutils.query import query_postgres
from tqdm import tqdm

import snowflake.connector
from sdp_snowflake_utils.oauth import SnowflakeOAuth


src_base = os.path.dirname(os.path.realpath(__file__))
config_file_path = os.path.abspath(os.path.join(src_base, "../config.json"))
with open(config_file_path, "r") as f:
    CONFIG_FILE = json.load(f)


def query_data():
    user = "rarevalo".upper()
    role = "SNOWFLAKE_SOFI_PRD_RISK_MGMT_HUB_RO".upper()
    warehouse = "TDM_RISK_MGMT_HUB"
    database = "TDM_RISK_MGMT_HUB"
    schema = "MODELED"

    credentials = SnowflakeOAuth.fetch(user, role)

    connection_parameters = {
        "account": "sofi",
        "authenticator": "oauth",
        "user": credentials.user,
        "role": credentials.role,
        "token": credentials.access_token,
        "warehouse": warehouse,
        "database": database,
        "schema": schema,
    }

    conn = snowflake.connector.connect(**connection_parameters)

    """
    Get raw data with one or more queries.
    """
    raw_data = {}

    for name, qinf in CONFIG_FILE["sql_query_files"].items():
        with open(os.path.join(src_base, qinf["query"])) as f:
            query = f.read()

        print("Running query {}.".format(name))
        cursor = conn.cursor()
        cursor.execute(query)

        raw_data[name] = df = cursor.fetch_pandas_all()

    print("Querying done")
    return raw_data


def process_raw_data(raw_data):
    """
    Process raw dataframes.
    """
    metadata_df = raw_data["metadata"]
    transactions_df = raw_data["transactions"]
    credit_df = raw_data["credit"]
    socure_df = raw_data["socure"]

    ### Dedup metadata
    print("Dedup metadata")
    metadata_df = metadata_df.sort_values(
        by=["borrower_id", "account_open_date"], ascending=False
    )
    metadata_df = metadata_df.drop_duplicates(
        subset=["borrower_id"]
    )  # keep most recently opened account.

    ### Convert dates to datetime.
    credit_df["credit_pull_date"] = pd.to_datetime(credit_df["credit_pull_date"])
    socure_df["socure_pull_date"] = pd.to_datetime(socure_df["socure_pull_date"])
    transactions_df["time_of_day"] = (
        transactions_df["time_of_day"].astype(str).replace("None", "00:00:00")
    )
    transactions_df["transaction_datetime"] = pd.to_datetime(
        (
            transactions_df["transaction_created_date_id"].astype(str)
            + transactions_df["time_of_day"]
        ),
        format="%Y%m%d%H:%M:%S",
        errors="coerce",
    )

    ### Join metadata, credit, and socure to transactions.
    # Join dataframes
    print("Join")
    transactions_df = pd.merge(
        transactions_df, metadata_df, how="inner", on="borrower_id"
    )

    # Sort by date for time sensitive merge.
    transactions_df = transactions_df.sort_values(by=["transaction_datetime"])
    credit_df = credit_df.sort_values(by=["credit_pull_date"])
    socure_df = socure_df.sort_values(by=["socure_pull_date"])

    credit_df = credit_df.dropna(subset=["credit_pull_date", "borrower_id"])
    socure_df = socure_df.dropna(subset=["socure_pull_date", "borrower_id"])
    transactions_df = pd.merge_asof(
        transactions_df,
        credit_df,
        left_on="transaction_datetime",
        right_on="credit_pull_date",
        by="borrower_id",
    )
    transactions_df = pd.merge_asof(
        transactions_df,
        socure_df,
        left_on="transaction_datetime",
        right_on="socure_pull_date",
        by="borrower_id",
    )

    # Delete raw dataframes to clear memory.
    del metadata_df
    del credit_df
    del socure_df
    gc.collect()

    ### Match deposits to returns.
    # Returned deposits defined as negative ACHDD or DWACHRET, DWCKCB.
    transactions_df["is_return"] = transactions_df["transaction_code"].isin(
        ["DWACHRET", "DWCKCB"]
    ) | (
        (transactions_df["transaction_code"] == "ACHDD")
        & (transactions_df["transaction_amount"] < 0)
    )

    def match_return_to_deposit(banking_transaction_details_id, transactions_df):
        """
        Match returns with their original transaction (within 5 business days).
        args:
            banking_transaction_details_id: transaction id of return
            transactions_df: DataFrame with all transactions
        returns:
            list of transactions ids that could match this return
        """
        transaction = transactions_df[
            transactions_df["banking_transaction_details_id"]
            == banking_transaction_details_id
        ]
        # reduce to transactions made by this borrower
        ttmp = transactions_df[
            transactions_df["borrower_id"].isin(transaction["borrower_id"])
        ]
        # transactions made on or before the return date
        ttmp = ttmp[
            ttmp["transaction_datetime"] <= transaction["transaction_datetime"].iloc[0]
        ]
        five_business_days_prior = transaction["transaction_datetime"].iloc[0] - BDay(5)
        # transaction created within 10 days of the return
        ttmp = ttmp[ttmp["transaction_datetime"] >= five_business_days_prior]
        # transactions with opposite value of deposit
        ttmp = ttmp[
            ttmp["transaction_amount"] == -1 * transaction["transaction_amount"].iloc[0]
        ]

        # ensure original transaction is in ['ACH Deposit', 'Add Money From Check']
        if transaction["transaction_code"].iloc[0] == "DWCKCB":
            ttmp = ttmp[ttmp["transaction_code"] == "DDCK"]
        else:
            ttmp = ttmp[ttmp["transaction_code"] == "ACHDD"]
        return ttmp["banking_transaction_details_id"]

    # Drop duplicate columns by name.
    transactions_df = transactions_df.loc[:, ~transactions_df.columns.duplicated()]

    transactions_df["is_returned"] = [0 for i in range(len(transactions_df))]
    for transaction_id in tqdm(
        transactions_df[transactions_df["is_return"]][
            "banking_transaction_details_id"
        ].values
    ):
        for idx in match_return_to_deposit(transaction_id, transactions_df).index:
            transactions_df.at[idx, "is_returned"] = 1

    ### Define target and indeterminate population.
    transactions_df["target"] = transactions_df["is_returned"]

    # Indeterminate deposits.
    transactions_df = pd.merge(
        transactions_df,
        transactions_df.groupby("borrower_id")["borrower_id"]
        .count()
        .rename("num_transactions_all_time")
        .to_frame(),
        how="left",
        on="borrower_id",
    )
    transactions_df["is_indeterminate"] = (
        (transactions_df["num_transactions_all_time"] < 3)
        | (transactions_df["target"] & (transactions_df["latest_acc_bal"] > 0))
        | (
            ~transactions_df["target"]
            & (
                transactions_df["account_closed_reason"].isin(
                    [
                        "Closed by SoFi - Charge-Off / Write-Off",
                        "Closed by SoFi - Risk Request",
                    ]
                )
                | (transactions_df["latest_acc_bal"] <= 0)
            )
        )
    )

    return {"processed": transactions_df}


def get_modeling_dataframe(processed_df):
    """
    Get modeling dataframe from processed data.
    """
    # Only keep ACH/Check depoits with positive amounts.
    processed_df = processed_df[
        (processed_df["transaction_code"].isin(["ACHDD", "DDCK"]))
        & (processed_df["transaction_amount"] > 0)
    ]

    # Remove indeterminate examples.
    processed_df = processed_df[~processed_df["is_indeterminate"]]

    # Remove accounts not opened between 2019-01-01 and 2019-06-30
    processed_df["account_open_date"] = pd.to_datetime(
        processed_df["account_open_date"]
    )
    processed_df = processed_df[
        processed_df["account_open_date"].between(
            pd.to_datetime("2019-01-01"), pd.to_datetime("2019-06-30")
        )
    ]

    # Remove SoFi employees
    processed_df = processed_df[~processed_df["sofi_employee_ind"]]
    print("Processed raw data")
    return processed_df


def save_dataframes(dataframes, base_path, prefix=None):
    """
    Takes a dictionary of dataframes with format:
        {name: dataframe}
    and saves them as base_path/prefix-name-timestamp.csv
    """
    timestamp_str = str(int(time.time()))

    if "data" not in CONFIG_FILE:
        CONFIG_FILE["data"] = {}

    if prefix not in CONFIG_FILE["data"]:
        CONFIG_FILE["data"][prefix] = {}

    if prefix:
        base_path = os.path.join(base_path, prefix)
    os.makedirs(base_path, exist_ok=True)

    for name, df in dataframes.items():
        fname = name + "_" + timestamp_str + ".feather"
        fpath = os.path.join(base_path, fname)

        _to_feather(df, fpath)

        CONFIG_FILE["data"][prefix][name] = os.path.join(prefix, fname)

    with open(config_file_path, "w") as f:
        json.dump(CONFIG_FILE, f)


def load_dataframes(prefix, base_path="data"):
    """
    Load dataframes from paths in config files.
    """
    dataframes = {}

    for name, path in CONFIG_FILE["data"][prefix].items():
        dataframes[name] = pd.read_feather(os.path.join(base_path, path))

    return dataframes


def _to_feather(df, path):
    """
    Handle bad formats.
    """
    for col in df.columns:
        if isinstance(df[col].iloc[0], datetime.time):
            df[col] = df[col].astype(str)

    df.reset_index().to_feather(path)


def _to_s3(prefix, base_path="data"):
    """
    Send local files to S3.
    """
    for name, path in CONFIG_FILE["data"][prefix].items():
        path = os.path.join(base_path, path)
        upload_s3(
            bucket_name=CONFIG_FILE["s3_bucket"],
            path_local=path,
            path_s3=os.path.join(CONFIG_FILE["s3_base_path"], path),
        )


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(
        description="Command line interface to data manipulation functions.",
        prefix_chars="-",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        default=False,
        help="Run end to end data pipeline.",
        dest="ga",
    )
    parser.add_argument(
        "-r",
        "--raw",
        action="store_true",
        default=False,
        help="Pull raw data files.",
        dest="gr",
    )
    parser.add_argument(
        "-p",
        "--processed",
        action="store_true",
        default=False,
        help="Build processed data file.",
        dest="gp",
    )
    parser.add_argument(
        "-f",
        "--features",
        type=str,
        default=None,
        help="Build features from processed data Pass path to processed data file.",
        dest="gf",
    )
    parser.add_argument(
        "-m",
        "--modeling",
        type=str,
        default=None,
        help="Convert processed data file to modeling data file. Pass path to processed data file.",
        dest="gm",
    )
    parser.add_argument(
        "-b",
        "--base_dir",
        type=str,
        default="data",
        help="Base directory to which files should be saved.",
        dest="bd",
    )
    parser.add_argument(
        "-u",
        "--upload_s3",
        type=str,
        default=None,
        help="Upload latest data files to S3. Can specify prefix.",
        dest="us",
    )
    args = parser.parse_args()

    if args.ga:
        # E2E data pipeline.
        pass
    elif args.gr:
        # Build raw data files.
        # Query for raw data files:
        raw_data = query_data()
        # Save raw data files:
        save_dataframes(raw_data, args.bd, prefix="raw")
    elif args.gp:
        # Build processed data file.
        # Query for raw data files:
        raw_data = load_dataframes(prefix="raw", base_path=args.bd)
        # Process raw data:
        processed_data = process_raw_data(raw_data)
        # Save processed data:
        save_dataframes(processed_data, args.bd, prefix="processed")
    elif args.gf is not None:
        # Build features.
        # Load dataframe:
        df = pd.read_feather(args.gf)
        # Build features:
        featurized_data = {"features": features.transform(df)}
        # Save data:
        save_dataframes(featurized_data, args.bd, prefix="features")
    elif args.gm is not None:
        # Build modeling dataframe.
        # Load dataframe:
        df = pd.read_feather(args.gm)
        # Build features:
        modeling_data = {"modeling": get_modeling_dataframe(df)}
        # Save data:
        save_dataframes(modeling_data, args.bd, prefix="modeling")

    if args.us is not None:
        # Upload data to S3
        if args.us == "all":
            pass  # TODO - upload all prefixes
        else:
            _to_s3(args.us)


if __name__ == "__main__":
    main()
