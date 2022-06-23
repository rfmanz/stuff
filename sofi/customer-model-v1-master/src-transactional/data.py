"""
Obtain and clean data.
"""

import features
import processing

import argparse
import datetime
import gc
import os
import time
import json

import pandas as pd

from multiprocessing import Pool, cpu_count

from rdsutils.aws import upload_s3
from rdsutils.query import query_postgres


src_base = os.path.dirname(os.path.realpath(__file__))
config_file_path = os.path.abspath(
    os.path.join(src_base, "../config-transactional.json")
)
with open(config_file_path, "r") as f:
    CONFIG_FILE = json.load(f)


def query_data(base_path="data", prefix=None):
    """
    Get raw data with one or more queries.
    """
    timestamp_str = str(int(time.time()))

    for name, qinf in CONFIG_FILE["sql_query_files"].items():
        with open(os.path.join(src_base, qinf["query"])) as f:
            query = f.read()

        print("Running query {}.".format(name))
        df = query_postgres(query=query, port=qinf["port"], database=qinf["database"])
        save_dataframes(
            {name: df}, base_path, prefix, timestamp_str, include_subdir=True
        )

        # clear memory when running several queries
        del df
        gc.collect()

    _set_config_file_field("data_pull_date", pd.datetime.today().strftime("%Y-%m-%d"))


def process_raw_data(base_path, prefix_in, prefix_out):
    """
    Process raw dataframes.
    """
    timestamp_str = str(int(time.time()))
    data_pull_date = pd.to_datetime(_get_config_file_field("data_pull_date"))

    # 1. Processing banking data
    banking_df = load_dataframe(prefix_in, "banking", base_path)

    banking_df = processing.process_banking(banking_df, data_pull_date=data_pull_date)

    save_dataframes(
        {"banking": banking_df},
        base_path,
        prefix_out,
        timestamp_str,
        include_subdir=True,
    )
    del banking_df
    gc.collect()

    # 2. Process customer data
    # Not yet implemented

    # 3. Process experian credit pull
    ecp_df = load_dataframe(prefix_in, "experian_credit_pull", base_path)

    ecp_df = processing.process_experian_credit_pull(ecp_df)

    save_dataframes(
        {"experian_credit_pull": ecp_df},
        base_path,
        prefix_out,
        timestamp_str,
        include_subdir=True,
    )
    del ecp_df
    gc.collect()

    # 4. Process giact
    giact_df = load_dataframe(prefix_in, "giact", base_path)

    giact_df = processing.process_giact(giact_df)

    save_dataframes(
        {"giact": giact_df}, base_path, prefix_out, timestamp_str, include_subdir=True
    )
    del giact_df
    gc.collect()

    # 5. Process quovo
#     quovo_df = load_dataframe(prefix_in, "quovo", base_path)

#     quovo_df = processing.process_quovo(quovo_df)

#     save_dataframes(
#         {"quovo": quovo_df}, base_path, prefix_out, timestamp_str, include_subdir=True
#     )
#     del quovo_df
#     gc.collect()

    # 6. Process socure data
    socure_df = load_dataframe(prefix_in, "socure", base_path)

    socure_df = processing.process_socure(socure_df)

    save_dataframes(
        {"socure": socure_df}, base_path, prefix_out, timestamp_str, include_subdir=True
    )
    del socure_df
    gc.collect()

    # 7. Process threat metrix data
#     tmx_df = load_dataframe(prefix_in, "threat_metrix", base_path)

#     tmx_df = processing.process_threat_metrix(tmx_df)

#     save_dataframes(
#         {"threat_metrix": tmx_df},
#         base_path,
#         prefix_out,
#         timestamp_str,
#         include_subdir=True,
#     )
#     del tmx_df
#     gc.collect()

    # 8. Process transactions data
    transactions_df = load_dataframe(prefix_in, "transactions", base_path)

    transactions_df = processing.process_transactions(transactions_df)

    save_dataframes(
        {"transactions": transactions_df},
        base_path,
        prefix_out,
        timestamp_str,
        include_subdir=True,
    )
    del transactions_df
    gc.collect()

    # 9. Process user metadata
    user_metadata_dw_df = load_dataframe(prefix_in, "user_metadata_dw", base_path)

    user_metadata_dw_df = processing.process_user_metadata_dw(user_metadata_dw_df)

    save_dataframes(
        {"user_metadata_dw": user_metadata_dw_df},
        base_path,
        prefix_out,
        timestamp_str,
        include_subdir=True,
    )
    del user_metadata_dw_df
    gc.collect()

    # 10. Process banking account restrictions
    banking_account_restrictions_df = load_dataframe(
        prefix_in, "banking_account_restrictions", base_path
    )

    banking_account_restrictions_df = processing.process_banking_account_restrictions(
        banking_account_restrictions_df
    )

    save_dataframes(
        {"banking_account_restrictions": banking_account_restrictions_df},
        base_path,
        prefix_out,
        timestamp_str,
        include_subdir=True,
    )
    del banking_account_restrictions_df
    gc.collect()


def join_processed_data(base_path, prefix_in, prefix_out):
    """
    Join processed data.
    """
    timestamp_str = str(int(time.time()))

    # 1. Get base df
    transactions_df = load_dataframe(prefix_in, "transactions", base_path)

    transactions_cols = [
        "borrower_id",
        "business_account_number",
        "transaction_datetime",
        "transaction_code",
        "transaction_comment",
        "is_return",
        "transaction_amount",
        "account_ending_balance",
        "card_present_ind",
        "external_account_number",
        "real_ending_balance",
    ]

    user_metadata_dw_df = load_dataframe(prefix_in, "user_metadata_dw", base_path)

    user_metadata_dw_cols = ["borrower_id", "user_id", "sofi_employee_ind"]

    base_df = pd.merge(
        transactions_df[transactions_cols],
        user_metadata_dw_df[user_metadata_dw_cols],
        how="inner",
        on="borrower_id",
    )

    del transactions_df
    del user_metadata_dw_df
    gc.collect()

    # 2. Merge updated banking data
    banking_df = load_dataframe(prefix_in, "banking", base_path)

    banking_cols = [
        "business_account_number",
        "odt",
        "dtc",
        "bal",
        "afdep",
        "dob",
        "mstate",
        "mzip",
        "closed_reason",
        "restricted_reason",
    ]

    base_df = pd.merge(
        base_df, banking_df[banking_cols], how="inner", on="business_account_number"
    )

    del banking_df
    gc.collect()

    # 2.5 Sort base df
    base_df = base_df.sort_values(by=["transaction_datetime"])

    # 3. Join customer data
    # Not yet implemented

    # 4. Join credit pull data
    ecp_df = load_dataframe(prefix_in, "experian_credit_pull", base_path)
    ecp_df = ecp_df.sort_values(by=["credit_pull_date"])

    ecp_cols = [
        "user_id",
        "credit_pull_date",
        "fico_score",
        "vantage_score",
        "all7120",
        "all8220",
        "bcc2800",
        "bcc7120",
        "bcx3423",
        "iln5520",
        "iqt9413",
        "iqt9415",
        "mtf5820",
        "stu5031",
        "credit_card_loan_amount",
        "delinquencies_90_days",
        "education_loan_amount",
        "mortgage_loan_amount",
        "secured_loan_amount",
        "total_outstanding_balance",
        "total_tradelines_open",
        "unsecured_loan_amount",
    ]

    base_df = pd.merge_asof(
        base_df,
        ecp_df[ecp_cols],
        left_on="transaction_datetime",
        right_on="credit_pull_date",
        by="user_id",
    )

    del ecp_df
    gc.collect()

    # 5. Join GIACT data
    giact_df = load_dataframe(prefix_in, "giact", base_path)
    giact_df = giact_df.sort_values(by=["giact_created_date"])

    giact_cols = [
        "business_account_number",
        "giact_created_date",
        "giact_first_link_date",
        "giact_last_link_date",
        "giact_is_pass",
        "giact_is_decline",
        "giact_is_other",
        "giact_nr_pass",
        "giact_nr_decline",
        "giact_nr_other",
    ]

    base_df = pd.merge_asof(
        base_df,
        giact_df[giact_cols],
        left_on="transaction_datetime",
        right_on="giact_created_date",
        by="business_account_number",
    )

    del giact_df
    gc.collect()

    # 6. Join quovo
#     quovo_df = load_dataframe(prefix_in, "quovo", base_path)
#     quovo_df = quovo_df.sort_values(by=["fetched_from_quovo_dt"])

#     quovo_cols = [
#         "user_id",
#         "fetched_from_quovo_dt",
#         "quovo_first_link_date",
#         "quovo_last_link_date",
#         "quovo_max_avail_bal",
#         "quovo_min_avail_bal",
#         "quovo_total_pos_bal",
#         "quovo_total_neg_bal",
#         "quovo_max_nr_transactions",
#         "quovo_min_nr_transactions",
#         "quovo_total_nr_transactions",
#         "quovo_largest_checking_or_savings_balance",
#         "quovo_nr_linked_checking_or_savings",
#     ]

#     base_df = pd.merge_asof(
#         base_df,
#         quovo_df[quovo_cols],
#         left_on="transaction_datetime",
#         right_on="fetched_from_quovo_dt",
#         by="user_id",
#     )

#     del quovo_df
#     gc.collect()

    # 7. Join socure data
    socure_df = load_dataframe(prefix_in, "socure", base_path)
    socure_df = socure_df.sort_values(by=["created_dt"])

    socure_cols = [
        "user_id",
        "created_dt",
        "fraud_score_1",
        "fraud_score_2",
        "address_risk_score",
        "email_risk_score",
        "phone_risk_score",
        "name_address_correlation",
        "name_email_correlation",
        "name_phone_correlation",
        "nr_social_profiles_found",
    ]

    base_df = pd.merge_asof(
        base_df,
        socure_df[socure_cols],
        left_on="transaction_datetime",
        right_on="created_dt",
        by="user_id",
    )

    del socure_df
    gc.collect()

#     # 8. Join threat metrix data
#     tmx_df = load_dataframe(prefix_in, "threat_metrix", base_path)
#     tmx_df = tmx_df.sort_values(by=["created_dt"])

#     tmx_cols = [
#         "user_id",
#         "tmx_created_dt",
#         "os",
#         "dns_ip_region",
#         "tmx_risk_rating",
#         "time_zone",
#         "page_time_on",
#     ]

#     base_df = pd.merge_asof(
#         base_df,
#         tmx_df,
#         left_on="transaction_datetime",
#         right_on="tmx_created_dt",
#         by="user_id",
#     )

#     del tmx_df
#     gc.collect()

    # 9. Join account restrictions data
    banking_account_restrictions_df = load_dataframe(
        prefix_in, "banking_account_restrictions", base_path
    )
    banking_account_restrictions_cols = [
        "business_account_number",
        "last_unrestricted_date",
        "first_restricted_by_risk_date",
    ]

    base_df = pd.merge(
        base_df,
        banking_account_restrictions_df[banking_account_restrictions_cols],
        how="inner",
        on="business_account_number",
    )

    del banking_account_restrictions_df
    gc.collect()

    save_dataframes({"base": base_df}, base_path, prefix_out, timestamp_str)


### Helper functions
def reverse_df(df):
    """ Helper for forward looking rolling function """
    # reverse dataset
    reverse_df = df.iloc[::-1]
    ri = reverse_df.index

    # re-reverse index
    reverse_df.index = ri[0] - ri + ri[-1]

    return reverse_df


## Need to define forward looking roll-ups at top level to parallelize.
def get_nr_returns_90d(df):
    return reverse_df(reverse_df(df).rolling("90d", min_periods=1)["is_return"].sum())


def get_bal_after_90d(df):
    return reverse_df(
        reverse_df(df)
        .rolling("90d", min_periods=1)["real_ending_balance"]
        .apply(lambda a: a[0], raw=True)
    )


def applyParallel(dfGrouped, func):
    """ Helper to parallelize apply over groupby """
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)


def stupid_hack(x):
    try:
        return x[-1]
    except:
        return x


def get_labels(df):
    """
    Get add labels to processed data.
    """
    data_pull_date = pd.to_datetime(_get_config_file_field("data_pull_date"))

    # sort data for roll-ups
    df = df.sort_values(by=["business_account_number", "transaction_datetime"])

    df["days_to_acc_close"] = (
        pd.to_datetime(df["dtc"]) - df["transaction_datetime"]
    ).dt.days
    df["account_closed_by_risk_in_next_90d"] = df["closed_reason"].isin(
        ["Closed by SoFi - Risk Request", "Closed by SoFi - Charge-Off / Write-Off"]
    ) & (df["days_to_acc_close"] <= 90)

    df["last_unrestricted_date_in_next_90d"] = (
        df["last_unrestricted_date"] - df["transaction_datetime"]
    ).dt.days.between(0, 90)

    # get most recent account balance
    df = pd.merge(
        df,
        df.groupby("business_account_number")["real_ending_balance"]
        .last()
        .rename("latest_acc_bal")
        .reset_index(),
        how="left",
        on="business_account_number",
    )

    # does account chg/wrt off in next 90 days?
    df["is_chg_wrt_off_in_90d"] = (
        df["chg_wrt_off_date"] - df["transaction_datetime"]
    ).dt.days <= 90

    # Set index to transaction datetime.
    df = df.set_index("transaction_datetime")

    # get num returns by borrower in the next 90 days
    df["nr_returns_in_next_90d"] = applyParallel(
        df[["business_account_number", "is_return"]].groupby("business_account_number"),
        get_nr_returns_90d,
    ).values

    # get this borrower's account balance after 90 days
    df["bal_after_90d"] = applyParallel(
        df[["business_account_number", "real_ending_balance"]].groupby(
            "business_account_number"
        ),
        get_bal_after_90d,
    ).values

    df = df.reset_index()

    # drop non-deposit types     
    #     df = df[df['transaction_code'].isin(deposit_data_types) & (df['transaction_amount'] > 0)]

    #     def get_target_1(df):  # for record keeping
    #         """
    #         Thomas' preferred target.

    #         BAD WHEN:
    #             - Account charges/writes off in next 30 days
    #             - Customer has return in next month AND
    #               does not end month with a positive balance
    #             - Customer ends month with a bal <= $-100
    #         INDETERMINATE WHEN:
    #             - Good AND account closed by risk in next month
    #             - GOOD AND account ends month with negative balance between
    #               $-99.99 and $-0.01
    #             - BAD AND balance after 60 days is positive

    #             ---- always ----
    #             - Transaction made less than a month before the data pull
    #               (no full time window)
    #             - Transaction made by a SoFi employee

    #         """
    #         df['target_1'] = (df['is_ever_chg_wrt_off'] > 0) |\
    #                          ((df['nr_returns_in_next_60d'] > 0) & (df['bal_after_60d'] <= 0)) |\
    #                          (df['bal_after_60d'] <= -100)
    #         df['indeterminate_1'] = (~df['is_ever_chg_wrt_off']) &\
    #                                 ((~df['target_1'] & df['account_closed_by_risk_in_next_90d']) |\
    #                                  (~df['target_1'] & df['bal_after_60d'].between(-99.99, -0.01)) |\
    #                                  (~df['has_30d_horizon']) | (df['sofi_employee_ind']) | \
    #                                  (df['target_1'] & (df['bal_after_60d'] > 0)) |\
    #                                  (df['target_1'] & (df['latest_acc_bal'] > 0)))
    #         return df

    #     def get_target_2(df):
    #         """
    #         Target used by Dan for evaluation:

    #         case when bf.current_account_balance<0 then 1
    #              when ba.account_restricted_reason like '%No%' then 1
    #              when ba.account_closed_reason like '%Risk%' then 1 else 0 end
    #         """
    #         df['target_2'] = (df['latest_acc_bal'] < 0) | \
    #                          (df['restricted_reason'].str.contains('No')) | \
    #                          (df['closed_reason'].str.contains('Risk'))

    #         return df

    #     def get_target_3(df):
    #         """
    #         Thomas take on Dan's target.
    #         """
    #         df['target_3'] = (df['is_ever_chg_wrt_off'] > 0) |\
    #                          ((df['nr_returns_all_time'] > 0) & (df['latest_acc_bal'] <= 0)) |\
    #                          (df['latest_acc_bal'] <= -5)
    #         df['indeterminate_3'] = (~df['is_ever_chg_wrt_off']) &\
    #                                 ((~df['target_1'] & df['closed_reason'].str.contains('Risk')) |\
    #                                  (~df['target_1'] & df['restricted_reason'].str.contains('No')) |\
    #                                  (~df['target_1'] & (df['latest_acc_bal'] < 0)) |\
    #                                  (~df['has_30d_horizon']) | (df['sofi_employee_ind']))
    #         return df

    #     df = get_target_1(df)
    #     df = get_target_2(df)
    #     df = get_target_3(df)
    
    df["target"] = get_target(df)
    df["indeterminate"] = get_indeterminate_dynamic(df)

    return df.reset_index()


def get_target(df):
    """
    BAD WHEN:
        - Account charges/writes off in next 90 days
        - Account closed by risk in next 90 days
        - Account restricted (no outbound trns/no trns activity) in next 90 days
        - Account balance after 90 days is negative
        - Account has an ACH or Check chargeback in next 90 days (and bal after 90 days is 0 or less)
    """
    targets = (
        (df["is_chg_wrt_off_in_90d"])
        | (df["account_closed_by_risk_in_next_90d"])
        | (
            df["last_unrestricted_date_in_next_90d"]
            & df["restricted_reason"].str.startswith("No")
        )
        | (df["bal_after_90d"] < 0)
        | ((df["nr_returns_in_next_90d"] > 0) & (df["bal_after_90d"] <= 0))
    )

    return targets


def get_indeterminate_dynamic(df):
    """
    indeterminate definitions

    1. bad but recovered account balance
    2. good but charged off
    3. good but recently down
    4. good but closed by risk
    5. good but restricted
    6. in-active
    """

    ind = ((df["target"] & (df["latest_acc_bal"] > 0)) 
           | (~df["target"] & (  # 1
        (~df["chg_wrt_off_date"].isna())  # 2
        | (df["latest_acc_bal"] < 0)  # 3
        | (
            df["closed_reason"].isin(  # 4
                [
                    "Closed by SoFi - Risk Request",
                    "Closed by SoFi - Charge-Off / Write-Off",
                ]
            )
        )
        | (df["restricted_reason"].str.startswith("No")))  # 5
    ))
    return ind

def save_dataframes(
    dataframes, base_path, prefix="", timestamp_str=None, include_subdir=False
):
    """
    Takes a dictionary of dataframes with format:
        {name: dataframe}
    and saves them as base_path/prefix-name-timestamp.feather
    """
    if timestamp_str is None:
        timestamp_str = str(int(time.time()))

    if "data" not in CONFIG_FILE:
        CONFIG_FILE["data"] = {}

    if prefix not in CONFIG_FILE["data"]:
        CONFIG_FILE["data"][prefix] = {}

    ppath = prefix
    if prefix:
        if len(dataframes) > 1 or include_subdir:
            ppath = os.path.join(prefix, prefix + "_" + timestamp_str)

    os.makedirs(os.path.join(base_path, ppath), exist_ok=True)

    for name, df in dataframes.items():
        fname = name + "_" + timestamp_str + ".feather"
        fpath = os.path.join(base_path, ppath, fname)

        _to_feather(df, fpath)

        CONFIG_FILE["data"][prefix][name] = os.path.join(ppath, fname)

    with open(config_file_path, "w") as f:
        json.dump(CONFIG_FILE, f, indent=4)


def load_dataframe(prefix, name, base_path="data"):
    """
    Load individual dataframe from path in config file.
    """
    path = CONFIG_FILE["data"][prefix][name]
    return pd.read_feather(os.path.join(base_path, path), use_threads=-1)


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

    if "level_0" in df.columns:
        df = df.drop("level_0", axis=1)

    df.reset_index(drop=True).to_feather(path)


def _get_config_file_field(field):
    """
    Get field value in config file.
    """
    return CONFIG_FILE[field]


def _set_config_file_field(field, val):
    """
    Set field value in config file.
    """
    CONFIG_FILE[field] = val

    with open(config_file_path, "w") as f:
        json.dump(CONFIG_FILE, f, indent=4)


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
        "-j",
        "--join",
        action="store_true",
        default=False,
        help="Join processed data.",
        dest="gj",
    )
    parser.add_argument(
        "-f",
        "--features",
        action="store_true",
        default=False,
        help="Build features from processed+joined data. Pass path to processed data file.",
        dest="gf",
    )
    parser.add_argument(
        "-l",
        "--labels",
        action="store_true",
        default=False,
        help="Add labels to processed/featurized dataframe.",
        dest="gl",
    )
    parser.add_argument(
        "-b",
        "--base_dir",
        type=str,
        default="data-transactional",
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
        # Not yet implemented.
        pass
    elif args.gr:
        # Build raw data files.
        # Query for raw data files:
        query_data(base_path=args.bd, prefix="raw")
    elif args.gp:
        # Build processed data files.
        # Process raw data:
        process_raw_data(base_path=args.bd, prefix_in="raw", prefix_out="processed")
    elif args.gj:
        # Join processed data files.
        join_processed_data(
            base_path=args.bd, prefix_in="processed", prefix_out="joined"
        )
    elif args.gf:
        # Build features.
        # Load dataframe:
        transactions_df = load_dataframe("joined", "base", args.bd)
        # Build features:
        featurized_data = {"features": features.transform(transactions_df)}
        # Save data:
        save_dataframes(featurized_data, args.bd, "features")
    elif args.gl:
        # Add labels.
        # Load dataframe:
        df = load_dataframe("features", "features", args.bd)
        # Add labels:
        labeled_data = {"labeled": get_labels(df)}
        # Save data:
        save_dataframes(labeled_data, args.bd, "labeled")

    if args.us is not None:
        # Upload data to S3
        if args.us == "all":
            pass  # TODO - upload all prefixes
        else:
            _to_s3(args.us)


if __name__ == "__main__":
    main()
