"""
Obtain and clean data.
"""

import combine
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

# from rdsutils.query import query_postgres

import snowflake.connector


src_base = os.path.dirname(os.path.realpath(__file__))
config_file_path = os.path.abspath(os.path.join(src_base, "../config.json"))


with open(config_file_path, "r") as f:
    CONFIG_FILE = json.load(f)


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


def query_data(base_path="data", prefix=None):
    """
    Get raw data with one or more queries.
    """
    timestamp_str = str(int(time.time()))

    for name, qinf in CONFIG_FILE["sql_query_files"].items():
        with open(os.path.join(src_base, qinf["query"])) as f:
            query = f.read()

        print("Running query {}.".format(name))
        df = run_query(query=query)
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

    # 1. Processing banking accounts data
    banking_accounts_df = load_dataframe(prefix_in, "banking_accounts", base_path)

    banking_accounts_df = processing.process_banking_accounts(
        banking_accounts_df, data_pull_date=data_pull_date
    )

    save_dataframes(
        {"banking_accounts": banking_accounts_df},
        base_path,
        prefix_out,
        timestamp_str,
        include_subdir=True,
    )
    del banking_accounts_df
    gc.collect()

    # 2. Process experian credit pull
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

    # 3. Process giact
    giact_df = load_dataframe(prefix_in, "giact", base_path)

    giact_df = processing.process_giact(giact_df)

    save_dataframes(
        {"giact": giact_df}, base_path, prefix_out, timestamp_str, include_subdir=True
    )
    del giact_df
    gc.collect()

    #     # 4. Process quovo
    #     quovo_df = load_dataframe(prefix_in, 'quovo', base_path)

    #     quovo_df = processing.process_quovo(quovo_df)

    #     save_dataframes({'quovo': quovo_df}, base_path,
    #                     prefix_out, timestamp_str, include_subdir=True)
    #     del quovo_df
    #     gc.collect()

    # 5. Process socure data
    socure_df = load_dataframe(prefix_in, "socure", base_path)

    socure_df = processing.process_socure(socure_df)

    save_dataframes(
        {"socure": socure_df}, base_path, prefix_out, timestamp_str, include_subdir=True
    )
    del socure_df
    gc.collect()

    # 6. Process threat metrix data
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

    # 7. Process transactions data
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

    # 8. Process user metadata
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


def join_processed_data(base_path, prefix_in, prefix_out, static_sample_dates=None):
    """
    Create sample of base_df, join the processed data.

    if static_sample_dates is None, randomly sample
    else, sample on the selected dates

    e.g.
    static_sample_dates = ['2019-03-15',
                      '2019-05-26',
                      '2019-08-07',
                      '2019-10-19',
                      '2019-12-31',
                      '2020-01-06',
                      '2020-02-17',
                      '2020-03-15',
                      '2020-04-15']

    """
    timestamp_str = str(int(time.time()))

    def sample_on_date_range(
        base_df,
        start="2019-03-01",
        end="2019-12-31",
        periods=40,
        static_sample_dates=None,
    ):
        """
        Helper fn to create records for each banking account on specific days.
        """
        df = base_df.copy()

        if static_sample_dates is None:
            dates = pd.date_range(
                pd.to_datetime(start), pd.to_datetime(end), periods=periods
            ).to_series()
        else:
            dates = static_sample_dates
        #             print('replicating TB modeling_df')
        #             dates = ['2019-03-15',
        #                      '2019-05-26',
        #                      '2019-08-07',
        #                      '2019-10-19',
        #                      '2019-12-31',
        #                      '2020-01-06',
        #                      '2020-02-17',
        #                      '2020-03-15',
        #                      '2020-04-15']
        print("sampling on dates: ", dates)

        dates = [pd.to_datetime(d) for d in dates]

        dfs = []

        for date in dates:
            df["sample_date"] = date
            dfs.append(df.copy())

        sampled_df = pd.concat(dfs, ignore_index=True)
        sampled_df = sampled_df[
            sampled_df["sample_date"] >= sampled_df["date_account_opened"]
        ]
        sampled_df = sampled_df[
            (sampled_df["sample_date"] < sampled_df["date_account_closed"])
            | sampled_df["date_account_closed"].isna()
        ]

        return sampled_df

    date_sample_start = pd.to_datetime(_get_config_file_field("date_sample_start"))
    date_sample_end = pd.to_datetime(_get_config_file_field("date_sample_end"))
    number_periods_in_sample = _get_config_file_field("number_periods_in_sample")

    # 0. Create base df by sampling from banking_accounts_df
    banking_accounts_df = load_dataframe(prefix_in, "banking_accounts", base_path)
    banking_accounts_df = banking_accounts_df.drop_duplicates(
        subset=["business_account_number"]
    )

    banking_accounts_cols = [
        "business_account_number",
        "user_id",
        "date_account_opened",
        "date_account_closed",
        "account_closed_reason",
        "account_age_days_asof_pull",
    ]

    base_df = sample_on_date_range(
        banking_accounts_df,
        date_sample_start,
        date_sample_end,
        number_periods_in_sample,
        static_sample_dates,
    )

    #     def temporary_sample_hack(base_df):
    #         """
    #         Temporary hack to sample account the day before they made an ACH or Check deposit (as requested by Dan).
    #         """
    #         d1 = pd.read_csv('~/SageMaker/money-risk-models/money-customer-risk/artifacts/dan_reqs/dan_file1.csv')
    #         d2 = pd.read_csv('~/SageMaker/money-risk-models/money-customer-risk/artifacts/dan_reqs/dan_file2.csv')
    #         d1['sample_date'] = pd.to_datetime(d1['transaction_posted_date_id'].astype(str), format='%Y%m%d')
    #         d2['sample_date'] = pd.to_datetime(d2['transaction_posted_date_id'].astype(str), format='%Y%m%d')
    #         mdf = pd.concat([d1, d2])
    #         base_df = pd.merge(mdf, base_df, how='left', on='business_account_number')
    #         return base_df
    #
    #     base_df = temporary_sample_hack(banking_accounts_df)
    #     print(len(base_df))

    del banking_accounts_df

    ### Joins

    # 1. Merge user metadata info.
    user_metadata_dw_df = load_dataframe(prefix_in, "user_metadata_dw", base_path)

    user_metadata_dw_cols = [
        "borrower_id",
        "user_id",
        "sofi_employee_ind",
        "date_of_birth",
    ]

    base_df = pd.merge(
        base_df, user_metadata_dw_df[user_metadata_dw_cols], how="left", on="user_id"
    )

    del user_metadata_dw_df
    gc.collect()
    print("1", len(base_df))

    # sort for time sensitive join (merge_asof)
    base_df = base_df.sort_values(by=["sample_date"])

    # 2. Merge transactions data.
    transactions_df = load_dataframe(prefix_in, "transactions", base_path)
    transactions_df = transactions_df.sort_values(by=["transaction_datetime"])

    #  Including all columns in transactions_df for now (because there are so many). Need to make
    # some type of config for this step (all of these in general.)
    transactions_cols = [
        "business_account_number",
        "endbal",
        "transaction_code",
        "transaction_amount",
        "external_account_number",
        "external_institution_id",
        "originating_company_id",
        "external_institution_trans_id",
        "originator_dfi_id",
        "merchant_name",
        "transaction_datetime",
        "real_ending_balance",
    ]

    #     roll_up_cols = ['sum_vault_transactions', 'has_chg_wrt_off_in_past_120d',
    #                     'nr_returns_in_past_30d', 'first_transaction_datetime',
    #                     'transaction_as_pct_of_balance', 'last_transaction_datetime',
    #                     'time_since_last_transaction', 'mean_trans_as_pct_of_bal',
    #                     'is_ach_return', 'nr_past_ach_returns', 'nr_ach_returns_3d',
    #                     'nr_ach_returns_10d', 'nr_ach_returns_30d', 'is_check_return',
    #                     'nr_past_check_returns', 'nr_check_returns_3d', 'nr_check_returns_10d',
    #                     'nr_check_returns_30d', 'nr_past_returns', 'nr_returns_30d', 'is_trans',
    #                     'is_deposit', 'nr_past_deposits', 'nr_deposits_3d', 'nr_deposits_10d',
    #                     'nr_deposits_30d', 'nr_past_transactions', 'nr_transactions_3d',
    #                     'nr_transactions_10d', 'nr_transactions_30d', 'pct_returned_deposits',
    #                     'pct_returned_deposits_30d', 'return_dollar_amount',
    #                     'dollar_val_returns', 'dollar_val_returns_3d', 'dollar_val_returns_10d',
    #                     'dollar_val_returns_30d', 'mean_account_balance_3d',
    #                     'mean_account_balance_10d', 'mean_account_balance_30d',
    #                     'std_account_balance_3d', 'std_account_balance_10d',
    #                     'std_account_balance_30d', 'deposit_transaction_amount',
    #                     'first_deposit_amount', 'last_deposit_amount', 'sum_deposits_3d',
    #                     'sum_deposits_10d', 'sum_deposits_30d', 'is_withdrawal',
    #                     'withdrawal_transaction_amount', 'sum_withdrawals_3d',
    #                     'sum_withdrawals_10d', 'sum_withdrawals_30d', 'mean_deposits_3d',
    #                     'mean_deposits_10d', 'mean_deposits_30d', 'mean_withdrawals_3d',
    #                     'mean_withdrawals_10d', 'mean_withdrawals_30d', 'max_deposits_3d',
    #                     'max_deposits_10d', 'max_deposits_30d', 'max_withdrawals_3d',
    #                     'max_withdrawals_10d', 'max_withdrawals_30d', 'is_dd',
    #                     'nr_direct_deposits', 'nr_transactions_all_time', 'ever_chg_wrt_off']

    base_df = pd.merge_asof(
        base_df,
        transactions_df[transactions_cols],
        left_on="sample_date",
        right_on="transaction_datetime",
        by="business_account_number",
    )
    print("2", len(base_df))
    del transactions_df
    gc.collect()

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
        left_on="sample_date",
        right_on="credit_pull_date",
        by="user_id",
    )
    print("3", len(base_df))

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
        left_on="sample_date",
        right_on="giact_created_date",
        by="business_account_number",
    )
    print("3", len(base_df))
    del giact_df
    gc.collect()

    # 6. Join quovo
    #     quovo_df = load_dataframe(prefix_in, 'quovo', base_path)
    #     quovo_df = quovo_df.sort_values(by=['fetched_from_quovo_dt'])

    #     quovo_cols = ['user_id', 'fetched_from_quovo_dt', 'quovo_first_link_date',
    #                   'quovo_last_link_date', 'quovo_max_avail_bal',
    #                   'quovo_min_avail_bal', 'quovo_total_pos_bal',
    #                   'quovo_total_neg_bal', 'quovo_max_nr_transactions',
    #                   'quovo_min_nr_transactions', 'quovo_total_nr_transactions',
    #                   'quovo_largest_checking_or_savings_balance', 'quovo_nr_linked_checking_or_savings']

    #     base_df = pd.merge_asof(base_df, quovo_df[quovo_cols], left_on='sample_date',
    #                             right_on='fetched_from_quovo_dt', by='user_id')

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
        left_on="sample_date",
        right_on="created_dt",
        by="user_id",
    )
    print("4", len(base_df))
    del socure_df
    gc.collect()

    # 8. Join threat metrix data
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
    #         base_df, tmx_df, left_on="sample_date", right_on="tmx_created_dt", by="user_id"
    #     )
    #     print("5", len(base_df))
    #     del tmx_df
    #     gc.collect()

    save_dataframes({"base": base_df}, base_path, prefix_out, timestamp_str)


def add_feature(df, fts, colname, default=None):
    # handles null during "groupby, rolling"
    fts = fts.rename(colname).reset_index()
    df = df.drop(columns=colname, errors="ignore")
    df = pd.merge(df, fts, how="left", on="business_account_number")
    df[colname] = df[colname].fillna(default)
    return df


def add_feature(modeling_df, fts, colname, default=None):
    fts = fts.rename(colname).reset_index()
    modeling_df = modeling_df.drop(columns=colname, errors="ignore")
    modeling_df = pd.merge(modeling_df, fts, how="left", on="business_account_number")
    modeling_df[colname] = modeling_df[colname].fillna(default)
    return modeling_df


def build_features_asof_date(
    transactions_df, modeling_df, sample_date=pd.to_datetime("2020-02-10")
):
    """
    Fn to build features asof a date.
    """
    print(1, len(modeling_df))
    # target stuffs
    modeling_df = pd.merge(
        modeling_df,
        transactions_df.groupby("business_account_number")[
            [
                "first_restricted_by_risk_date",
                "last_unrestricted_date",
                "dtc",
                "latest_acc_bal",
                "closed_reason",
                "restricted_reason",
                "chg_wrt_off_date",
                "first_transaction_datetime",
            ]
        ]
        .last()
        .reset_index(),
        how="left",
        on="business_account_number",
    )
    print(2, len(modeling_df))
    # pass it sorted for speed
    # transactions_df = transactions_df.sort_values(by=['business_account_number', 'transaction_datetime'])

    # more target stuffs
    tdf_offset = transactions_df[
        transactions_df["transaction_datetime"].between(
            sample_date, sample_date + datetime.timedelta(days=90)
        )
    ]
    modeling_df = add_feature(
        modeling_df,
        tdf_offset.groupby("business_account_number")["real_ending_balance"].last(),
        "bal_after_90d",
        modeling_df["real_ending_balance"],
    )
    modeling_df = add_feature(
        modeling_df,
        tdf_offset.groupby("business_account_number")["is_return"].sum(),
        "nr_returns_in_next_90d",
        0,
    )
    modeling_df = add_feature(
        modeling_df,
        tdf_offset.groupby("business_account_number")[
            "business_account_number"
        ].count(),
        "nr_transactions_next_90d",
        0,
    )

    tdf_offset = transactions_df[
        transactions_df["transaction_datetime"].between(
            sample_date, sample_date + datetime.timedelta(days=60)
        )
    ]
    modeling_df = add_feature(
        modeling_df,
        tdf_offset.groupby("business_account_number")[
            "business_account_number"
        ].count(),
        "nr_transactions_next_60d",
        0,
    )

    print(3, len(modeling_df))

    # 30d features
    transactions_df = transactions_df[
        transactions_df["transaction_datetime"].between(
            sample_date - datetime.timedelta(days=30), sample_date
        )
    ]
    modeling_df = add_feature(
        modeling_df,
        transactions_df.groupby("business_account_number")[
            "real_ending_balance"
        ].mean(),
        "mean_account_balance_30d",
        modeling_df["real_ending_balance"],
    )
    modeling_df = add_feature(
        modeling_df,
        transactions_df.groupby("business_account_number")[
            "withdrawal_transaction_amount"
        ].min(),
        "max_withdrawals_30d",
        0,
    )
    modeling_df = add_feature(
        modeling_df,
        transactions_df.groupby("business_account_number")[
            "deposit_transaction_amount"
        ].max(),
        "max_deposits_30d",
        0,
    )
    modeling_df = add_feature(
        modeling_df,
        transactions_df.groupby("business_account_number")["is_return"].sum(),
        "nr_returns_30d",
        0,
    )
    modeling_df = add_feature(
        modeling_df,
        transactions_df.groupby("business_account_number")[
            "business_account_number"
        ].count(),
        "nr_transactions_30d",
        0,
    )

    print(4, len(modeling_df))

    # 10d features
    transactions_df = transactions_df[
        transactions_df["transaction_datetime"].between(
            sample_date - datetime.timedelta(days=10), sample_date
        )
    ]
    modeling_df = add_feature(
        modeling_df,
        transactions_df.groupby("business_account_number")[
            "deposit_transaction_amount"
        ].sum(),
        "sum_deposits_10d",
        0,
    )

    # 3d features
    transactions_df = transactions_df[
        transactions_df["transaction_datetime"].between(
            sample_date - datetime.timedelta(days=3), sample_date
        )
    ]
    modeling_df = add_feature(
        modeling_df,
        transactions_df.groupby("business_account_number")[
            "return_dollar_amount"
        ].sum(),
        "dollar_val_returns_3d",
        0,
    )
    modeling_df = add_feature(
        modeling_df,
        transactions_df.groupby("business_account_number")[
            "business_account_number"
        ].count(),
        "nr_transactions_3d",
        0,
    )
    modeling_df = add_feature(
        modeling_df,
        transactions_df.groupby("business_account_number")[
            "real_ending_balance"
        ].mean(),
        "mean_account_balance_3d",
        modeling_df["real_ending_balance"],
    )

    # ratio features
    modeling_df["nr_trans_ratio"] = (
        modeling_df["nr_transactions_3d"] / modeling_df["nr_transactions_30d"]
    )
    modeling_df["bal_ratio"] = (
        modeling_df["mean_account_balance_3d"] / modeling_df["mean_account_balance_30d"]
    )
    modeling_df["days_since_first_deposit"] = (
        modeling_df["sample_date"] - modeling_df["first_transaction_datetime"]
    ).dt.days

    print("end", len(modeling_df))

    return modeling_df


def get_labels(df_sampled, transactions_df):

    sampled_dataframes = {}

    for date in df_sampled["sample_date"].unique():
        date_str = str(date).split()[0]
        dtm = df_sampled[df_sampled["sample_date"] == date]
        sampled_dataframes[date_str] = dtm.copy()

    transactions_df = transactions_df.sort_values(
        by=["business_account_number", "transaction_datetime"]
    )

    for dt, df in sampled_dataframes.items():
        df = build_features_asof_date(transactions_df, df, pd.to_datetime(dt))
        sampled_dataframes[dt] = df
        _ = gc.collect()

    sampled_df = pd.concat(sampled_dataframes.values(), axis=0)
    del sampled_dataframes
    gc.collect()

    sampled_df = sampled_df.sort_values(by=["sample_date"])
    transactions_df = transactions_df.sort_values(by=["transaction_datetime"])
    sampled_df = pd.merge_asof(
        sampled_df,
        transactions_df[
            [
                "business_account_number",
                "transaction_datetime",
                "transaction_as_pct_of_balance",
                "pct_returned_deposits",
                "time_since_last_transaction",
                "nr_past_transactions",
                "giact_time_since_last_link",
                "nr_direct_deposits",
                "dollar_val_dd",
                "lag_acc_open_first_transaction",
                "first_deposit_amount",
                "dollar_val_returns",
            ]
        ],
        left_on="sample_date",
        right_on="transaction_datetime",
        by="business_account_number",
    )

    sampled_df = sampled_df.drop(
        columns=[col for col in sampled_df.columns if "_y" in col]
    )
    sampled_df = sampled_df.rename(
        columns={col: col[:-2] for col in sampled_df.columns if "_x" in col}
    )

    sampled_df["is_chg_wrt_off_in_90d"] = (
        sampled_df["chg_wrt_off_date"] - sampled_df["sample_date"]
    ).dt.days <= 90
    sampled_df["days_to_acc_close"] = (
        pd.to_datetime(sampled_df["dtc"]) - sampled_df["sample_date"]
    ).dt.days
    sampled_df["account_closed_by_risk_in_next_90d"] = sampled_df["closed_reason"].isin(
        ["Closed by SoFi - Risk Request", "Closed by SoFi - Charge-Off / Write-Off"]
    ) & (sampled_df["days_to_acc_close"] <= 90)
    sampled_df["last_unrestricted_date_in_next_90d"] = (
        sampled_df["last_unrestricted_date"] - sampled_df["transaction_datetime"]
    ).dt.days.between(0, 90)
    sampled_df["is_chg_wrt_off_in_90d"] = (
        sampled_df["chg_wrt_off_date"] - sampled_df["sample_date"]
    ).dt.days <= 90

    #     def get_target(df):   # kept this for record keeping
    #         """
    #         BAD WHEN:
    #             - Account charges/writes off in next 90 days
    #             - Account closed by risk in next 90 days
    #             - Account restricted (no outbound trns/no trns activity) in next 90 days
    #             - Account balance after 90 days is negative
    #             - Account has an ACH or Check chargeback in next 90 days (and bal after 90 days is 0 or less)
    #         """
    #         df["target"] = (
    #             (df["is_chg_wrt_off_in_90d"])
    #             | (df["account_closed_by_risk_in_next_90d"])
    #             | (
    #                 df["last_unrestricted_date_in_next_90d"]
    #                 & df["restricted_reason"].str.startswith("No")
    #             )
    #             | (df["bal_after_90d"] < 0)
    #             | ((df["nr_returns_in_next_90d"] > 0) & (df["bal_after_90d"] <= 0))
    #         )

    #         df["target_with_restriction"] = (
    #             (df["is_chg_wrt_off_in_90d"])
    #             | (df["account_closed_by_risk_in_next_90d"])
    #             | (
    #                 df["last_unrestricted_date_in_next_90d"]
    #                 & df["restricted_reason"].str.startswith("No")
    #             )
    #             | (df["bal_after_90d"] < 0)
    #             | ((df["nr_returns_in_next_90d"] > 0) & (df["bal_after_90d"] <= 0))
    #         )

    #         df["indeterminate"] = (  # why is it defined like this?
    #             (pd.to_datetime(df["dtc"]) <= df["sample_date"])
    #             | (df["last_unrestricted_date"] <= df["sample_date"])
    #             | (df["chg_wrt_off_date"] <= df["sample_date"])
    #             | (df["target"] & (df["latest_acc_bal"] > 0))
    #             | (
    #                 ~df["target"] & (~df["chg_wrt_off_date"].isna())
    #                 | (df["latest_acc_bal"] < 0)
    #                 | (
    #                     df["closed_reason"].isin(
    #                         [
    #                             "Closed by SoFi - Risk Request",
    #                             "Closed by SoFi - Charge-Off / Write-Off",
    #                         ]
    #                     )
    #                 )
    #                 | (df["restricted_reason"].str.startswith("No"))
    #             )
    #         )

    #         df["indeterminate"] = (  # why is it defined like this?
    #             (df["nr_transactions_next_60d"] == 0)
    #             | (pd.to_datetime(df["dtc"]) <= df["sample_date"])
    #             | (df["last_unrestricted_date"] <= df["sample_date"])
    #             | (df["chg_wrt_off_date"] <= df["sample_date"])
    #             | (df["target"] & (df["latest_acc_bal"] > 0))
    #             | (
    #                 ~df["target"] & (~df["chg_wrt_off_date"].isna())
    #                 | (df["latest_acc_bal"] < 0)
    #                 | (
    #                     df["closed_reason"].isin(
    #                         [
    #                             "Closed by SoFi - Risk Request",
    #                             "Closed by SoFi - Charge-Off / Write-Off",
    #                         ]
    #                     )
    #                 )
    #                 | (df["restricted_reason"].str.startswith("No"))
    #             )
    #         )

    # add indeterminate_reason_components and recreated indeterminate

    #         return df

    sampled_df["target"] = get_target(sampled_df)
    sampled_df["indeterminate"] = get_indeterminate_static(sampled_df)

    return sampled_df


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


def get_indeterminate_static(df):
    """
    indeterminate definitions

    1. bad but recovered account balance
    2. good but charged off
    3. good but recently down
    4. good but closed by risk
    5. good but restricted
    6. in-active
    """

    ind = (
        (df["nr_transactions_next_60d"] == 0)  # 6
        | (df["target"] & (df["latest_acc_bal"] > 0))  # 1
        | (
            ~df["target"]
            & (
                ~df["chg_wrt_off_date"].isna()  # 2
                | (df["latest_acc_bal"] < 0)  # 3
                | df["restricted_reason"].str.startswith("No")  # 5
                | df["closed_reason"].isin(  # 4
                    [
                        "Closed by SoFi - Risk Request",
                        "Closed by SoFi - Charge-Off / Write-Off",
                    ]
                )
            )
        )
    )
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


def load_dataframe(prefix, name, base_path="data", debug=False):
    """
    Load individual dataframe from path in config file.
    """

    path = CONFIG_FILE["data"][prefix][name]
    if base_path == "data-transactional":
        # load from data-transactional folder
        # stupid hack...for this dynamic and static combination
        config_file_path = os.path.abspath(
            os.path.join(src_base, "../config-transactional.json")
        )
        with open(config_file_path, "r") as f:
            config_file = json.load(f)
        path = config_file["data"][prefix][name]

    df = pd.read_feather(os.path.join(base_path, path), use_threads=-1)
    if debug:
        df = df.iloc[:1000]
    return df


def load_dataframes(prefix, base_path="data", debug=False):
    """
    Load dataframes from paths in config files.
    """
    dataframes = {}

    for name, path in CONFIG_FILE["data"][prefix].items():
        dataframes[name] = pd.read_feather(os.path.join(base_path, path))
        if debug:
            dataframes[name] = dataframes[name].iloc[:1000]

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
        "--sample_and_join",
        action="store_true",
        default=False,
        help="Sample the base dataframe, join processed data.",
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
        "-c",
        "--combine",
        action="store_true",
        default=False,
        help="Process and combine the dynamic and static data",
        dest="gc",
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
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="""Enter debugging mode by only process first 1000 rows of data. 
                       Note all data still need to be loaded as of now due to feather format""",
        dest="debug",
    )
    parser.add_argument(
        "-af",
        "--all_features",
        action="store_true",
        default=False,
        help="""for combine, return all features calculated if true""",
        dest="all_features",
    )
    parser.add_argument(
        "-nf",
        "--no_filtering",
        action="store_false",
        default=True,
        help="""for combine, return all rows without filtering indeterminates""",
        dest="no_filtering",
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
        static_sample_dates = CONFIG_FILE["static_sample_dates"]
        join_processed_data(
            base_path=args.bd,
            prefix_in="processed",
            prefix_out="joined",
            static_sample_dates=static_sample_dates,
        )
    elif args.gf:
        raise NotImplemented("feature engineering is included in the labeling stage")
    #         # Build features.
    #         # Load dataframe:
    #         transactions_df = load_dataframe('joined', 'base', args.bd, args.debug)
    #         # Build features:
    #         featurized_data = {'features': features.transform(transactions_df)}
    #         # Save data:
    #         save_dataframes(featurized_data, args.bd,
    #                         'features')
    elif args.gl:
        # Add labels.
        # Load dataframe:
        df = load_dataframe("joined", "base", args.bd, args.debug)
        transactions_df = load_dataframe(
            "labeled", "labeled", "data-transactional", args.debug
        )
        # Add labels:
        labeled_data = {"labeled": get_labels(df, transactions_df)}
        # Save data:
        save_dataframes(labeled_data, args.bd, "labeled")
    elif args.gc:
        # combine data
        # dynamic is from src-transactional
        # static from src
        dynamic_full = load_dataframe(
            "labeled", "labeled", "data-transactional", args.debug
        )
        static_full = load_dataframe("labeled", "labeled", "data", args.debug)
        combined_data = {
            "combined": combine.combine_data(
                dynamic_full,
                static_full,
                CONFIG_FILE["date_sample_start"],
                CONFIG_FILE["date_sample_end"],
                args.all_features,
                args.no_filtering,
            )
        }
        save_dataframes(
            combined_data,
            args.bd,
            "combined" if not args.all_features else "combined_all_features",
        )

    if args.us is not None:
        # Upload data to S3
        if args.us == "all":
            pass  # TODO - upload all prefixes
        else:
            _to_s3(args.us)


if __name__ == "__main__":
    main()
