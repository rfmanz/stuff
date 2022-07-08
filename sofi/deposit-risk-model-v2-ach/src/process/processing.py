"""
Functions for processing raw data files.

PLEASE Do the following:
1. PREFIX ALL FUNCTION NAMES with `process_`
"""

import pandas as pd
import numpy as np
import src.utils as utils


def process_banking(banking_df, **kwargs):
    """
    Process banking data.
    """

    print("Processing banking data.")
    data_pull_date = kwargs["data_pull_date"]

    # banking_df["user_id"] = banking_df["zsofiid"]
    banking_df["acct_open_date"] = pd.to_datetime(banking_df["acct_open_date"]).dt.tz_localize(None)
    banking_df["date_of_birth"] = pd.to_datetime(banking_df["date_of_birth"]).dt.tz_localize(None)

    banking_df["account_age_days_asof_pull"] = (
        data_pull_date - banking_df["acct_open_date"]
    ).dt.days
    banking_df["customer_age_days_asof_pull"] = (
        data_pull_date - banking_df["acct_open_date"]
    ).dt.days

    banking_df = utils.add_prefix(
        banking_df,
        "banking_",
        exclusion=[
            "user_id",
            "borrower_id",
            "business_account_number",
            "transaction_datetime",
        ],
    )
    return banking_df


def process_experian_credit_pull(ecp_df, **kwargs):
    """
    Process experian credit pull data.
    """
    print("Processing experian credit pull data.")
    ecp_df["credit_pull_date"] = pd.to_datetime(ecp_df["credit_pull_date"])

    float_conv = [
        "all7120",
        "all8220",
        "bcc2800",
        "bcc7120",
        "bcx3423",
        "iln5520",
        "iqt9415",
        "iqt9413",
        "mtf5820",
        "stu5031",
        "delinquencies_90_days",
    ]
    ecp_df[float_conv] = ecp_df[float_conv].astype(float)
    ecp_df = utils.add_prefix(
        ecp_df,
        "ecp_",
        exclusion=[
            "user_id",
            "borrower_id",
            "business_account_number",
            "transaction_datetime",
        ],
    )
    return ecp_df


def process_giact(giact_df, **kwargs):
    """
    Process giact data.
    """
    print("Processing giact data.")
    giact_df = giact_df.dropna(subset=["user_id", "business_account_number"])
    giact_df["user_id"] = giact_df["user_id"].astype(int)
    giact_df["business_account_number"] = giact_df["business_account_number"].astype(
        int
    )
    giact_df = giact_df.dropna(subset=["giact_created_date"])
    giact_df["giact_created_date"] = pd.to_datetime(
        giact_df["giact_created_date"], unit="ms"
    ).dt.tz_localize(None)
    giact_df = giact_df.sort_values(by=["giact_created_date", "user_id"])

    giact_df["giact_first_link_date"] = giact_df.groupby("user_id")[
        "giact_created_date"
    ].cummin()
    giact_df["giact_last_link_date"] = giact_df.groupby("user_id")[
        "giact_created_date"
    ].cummax()

    giact_df["giact_is_pass"] = giact_df["giact_verification_response"].isin(
        ["PASS", "ACCEPT_WITH_RISK"]
    )
    giact_df["giact_is_decline"] = ~giact_df["giact_verification_response"].isin(
        ["PASS", "ACCEPT_WITH_RISK", "NO_DATA", "ERROR"]
    )
    giact_df["giact_is_other"] = giact_df["giact_verification_response"].isin(
        ["NO_DATA", "ERROR"]
    )

    giact_df["giact_nr_pass"] = giact_df.groupby("user_id")["giact_is_pass"].cumsum()
    giact_df["giact_nr_decline"] = giact_df.groupby("user_id")[
        "giact_is_decline"
    ].cumsum()
    giact_df["giact_nr_other"] = giact_df.groupby("user_id")["giact_is_other"].cumsum()

    giact_df = utils.add_prefix(
        giact_df,
        "giact_",
        exclusion=[
            "user_id",
            "borrower_id",
            "business_account_number",
            "transaction_datetime",
        ],
    )
    return giact_df


def process_socure(socure_df, **kwargs):
    """
    Process socure data.
    """
    print("Processing socure data.")
    socure_scores = [
        "fraud_score_1",
        "fraud_score_2",
        "address_risk_score",
        "email_risk_score",
        "phone_risk_score",
        "name_address_correlation",
        "name_email_correlation",
        "name_phone_correlation",
    ]
    socure_df = socure_df.dropna(subset=["user_id"])
    socure_df["created_dt"] = pd.to_datetime(socure_df["created_dt"]).dt.tz_localize(
        None
    )
    socure_df["user_id"] = socure_df["user_id"].astype(int)

    socure_df[socure_scores] = socure_df[socure_scores].astype(float)
    socure_df["nr_social_profiles_found"] = (
        socure_df["social_profiles_found"].str.count('"') / 2
    )

    socure_df = utils.add_prefix(
        socure_df,
        "socure_",
        exclusion=[
            "user_id",
            "borrower_id",
            "business_account_number",
            "transaction_datetime",
        ],
    )
    return socure_df


def process_threat_metrix(tmx_df, **kwargs):
    """
    Process threat metrix data.
    """
    print("Processing threat metrix data.")
    str_cols = [
        "os",
        "dns_ip_geo",
        "enabled_ck",
        "enabled_fl",
        "enabled_im",
        "enabled_js",
        "screen_res",
        "agent_brand",
        "device_name",
        "dns_ip_region",
        "agent_language",
        "tmx_risk_rating",
        "browser_language",
    ]
    int_cols = ["time_zone", "page_time_on"]

    tmx_df = tmx_df.dropna(subset=["user_id"])
    tmx_df["tmx_created_dt"] = pd.to_datetime(tmx_df["created_dt"]).dt.tz_localize(None)

    for col in str_cols + int_cols:
        tmx_df[col] = tmx_df[col].str.strip('[""]')
    tmx_df[int_cols] = tmx_df[int_cols].astype(float)
    tmx_df["user_id"] = tmx_df["user_id"].astype(int)

    tmx_df = utils.add_prefix(
        tmx_df,
        "tmx_",
        exclusion=[
            "user_id",
            "borrower_id",
            "business_account_number",
            "transaction_datetime",
        ],
    )
    return tmx_df


def process_transactions(transactions_df, **kwargs):
    """
    Process transactions data.
    """
    print("Processing transactions data.")
    transactions_df["time"] = (
        transactions_df["time"].astype(str).replace("None", "00:00:00")
    )
    transactions_df["transaction_datetime"] = pd.to_datetime(
        (transactions_df["tjd"].astype(str) + " " + transactions_df["time"]),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    ).dt.tz_localize(None)

    transactions_df["transaction_amount"] = transactions_df[
        "transaction_amount"
    ].astype(float)
    transactions_df["tamt_multiplier"] = np.where(transactions_df["is_credit"], 1, -1)
    transactions_df["transaction_amount"] = (
        transactions_df["transaction_amount"] * transactions_df["tamt_multiplier"]
    )

    # Returned deposits defined as negative ACHDD or DWACHRET, DWCKCB.
    transactions_df["is_return"] = transactions_df["transaction_code"].isin(
        ["DWACHRET", "DWCKCB"]
    ) | (
        (transactions_df["transaction_code"] == "ACHDD")
        & (transactions_df["transaction_amount"] < 0)
    )

    # This will change when we pull from a different source. Define external account number.
    #     transactions_df['external_account_number'] = transactions_df['external_account_number'].replace('', np.nan)

    # Correct account ending balance for accounts with vaults.
    vault_transaction_codes = ["DCWIVAULT", "DWTRF", "OTOD", "DWINT"]
    transactions_df = transactions_df.sort_values(
        by=["business_account_number", "transaction_datetime"]
    )
    transactions_df["vault_trans_amt"] = transactions_df[
        "transaction_amount"
    ] * transactions_df["transaction_code"].isin(
        vault_transaction_codes
    )  # transaction amount 0 unless vault transaction
    transactions_df["sum_vault_transactions"] = (
        transactions_df.groupby("business_account_number")["vault_trans_amt"]
        .cumsum()
        .values
    )
    transactions_df["real_ending_balance"] = (
        transactions_df["endbal"] - transactions_df["sum_vault_transactions"]
    )

    transactions_df = utils.add_prefix(
        transactions_df,
        "trnx_",
        exclusion=[
            "user_id",
            "borrower_id",
            "business_account_number",
            "transaction_datetime",
            "tseq",
        ],
    )
    return transactions_df


def process_user_metadata_dw(user_metadata_dw_df, **kwargs):
    """
    Process user metadata from dw.
    """
    print("Processing user metadata from dw.")

    user_metadata_dw_df = utils.add_prefix(
        user_metadata_dw_df,
        "user_meta_",
        exclusion=[
            "user_id",
            "borrower_id",
            "business_account_number",
            "transaction_datetime",
        ],
    )
    return user_metadata_dw_df


def process_banking_account_restrictions(banking_account_restrictions_df, **kwargs):
    """"""
    print("Processing banking account restrictions.")
    banking_account_restrictions_df["last_unrestricted_date"] = pd.to_datetime(
        banking_account_restrictions_df["last_unrestricted_date"]
    )
    banking_account_restrictions_df["first_restricted_by_risk_date"] = pd.to_datetime(
        banking_account_restrictions_df["first_restricted_by_risk_date"]
    )
    banking_account_restrictions_df = utils.add_prefix(
        banking_account_restrictions_df,
        "bk_acct_rstr_",
        exclusion=[
            "user_id",
            "borrower_id",
            "business_account_number",
            "transaction_datetime",
        ],
    )
    return banking_account_restrictions_df


def process_plaid(plaid_df, **kwargs):
    """"""
    print("Processing plaid.")
    plaid_df["user_id"] = plaid_df["user_id"].astype(int)  # this may throw an error

    plaid_df["available_bal"] = plaid_df["available_bal"].astype(float)
    plaid_df["current_bal"] = plaid_df["current_bal"].astype(float)

    plaid_df["current_as_of_dt"] = pd.to_datetime(
        plaid_df["current_as_of_dt"]
    ).dt.tz_localize(None)

    plaid_df = plaid_df.sort_values(by=["current_as_of_dt", "user_id"])

    plaid_df["pos_bal"] = plaid_df["available_bal"] * (plaid_df["available_bal"] > 0)
    plaid_df["neg_bal"] = plaid_df["available_bal"] * (plaid_df["available_bal"] < 0)

    plaid_df["plaid_first_link_date"] = plaid_df.groupby("user_id")[
        "current_as_of_dt"
    ].cummin()
    plaid_df["plaid_last_link_date"] = plaid_df.groupby("user_id")[
        "current_as_of_dt"
    ].cummax()

    plaid_df["plaid_max_avail_bal"] = plaid_df.groupby("user_id")[
        "available_bal"
    ].cummax()
    plaid_df["plaid_min_avail_bal"] = plaid_df.groupby("user_id")[
        "available_bal"
    ].cummin()

    plaid_df["plaid_total_pos_bal"] = plaid_df.groupby("user_id")["pos_bal"].cumsum()
    plaid_df["plaid_total_neg_bal"] = plaid_df.groupby("user_id")["neg_bal"].cumsum()

    plaid_df["is_checking_or_savings"] = plaid_df["account_type"].isin(
        ["Checking", "Savings"]
    )
    plaid_df.loc[
        plaid_df["is_checking_or_savings"], "available_balance_if_checking_or_savings"
    ] = plaid_df.loc[plaid_df["is_checking_or_savings"], "available_bal"]
    plaid_df["plaid_largest_checking_or_savings_balance"] = plaid_df.groupby("user_id")[
        "available_balance_if_checking_or_savings"
    ].cummax()
    plaid_df["plaid_available_bal"] = plaid_df["available_bal"]

    plaid_df = utils.add_prefix(
        plaid_df,
        "plaid_",
        exclusion=[
            "user_id",
            "borrower_id",
            "business_account_number",
            "transaction_datetime",
        ],
    )
    return plaid_df

# DB deprecated
# def process_quovo(quovo_df, **kwargs):
#     """
#     Process quovo data.
#     """
#     print("Processing quovo.")
#     quovo_df["user_id"] = quovo_df["user_id"].astype(int)  # this may throw an error

#     quovo_df["available_bal"] = quovo_df["available_bal"].astype(float)
#     quovo_df["current_bal"] = quovo_df["current_bal"].astype(float)

#     quovo_df["current_as_of_dt"] = pd.to_datetime(
#         quovo_df["current_as_of_dt"]
#     ).dt.tz_localize(None)

#     quovo_df = quovo_df.sort_values(by=["current_as_of_dt", "user_id"])

#     quovo_df["pos_bal"] = quovo_df["available_bal"] * (quovo_df["available_bal"] > 0)
#     quovo_df["neg_bal"] = quovo_df["available_bal"] * (quovo_df["available_bal"] < 0)

#     quovo_df["quovo_first_link_date"] = quovo_df.groupby("user_id")[
#         "current_as_of_dt"
#     ].cummin()
#     quovo_df["quovo_last_link_date"] = quovo_df.groupby("user_id")[
#         "current_as_of_dt"
#     ].cummax()

#     quovo_df["quovo_max_avail_bal"] = quovo_df.groupby("user_id")[
#         "available_bal"
#     ].cummax()
#     quovo_df["quovo_min_avail_bal"] = quovo_df.groupby("user_id")[
#         "available_bal"
#     ].cummin()

#     quovo_df["quovo_total_pos_bal"] = quovo_df.groupby("user_id")["pos_bal"].cumsum()
#     quovo_df["quovo_total_neg_bal"] = quovo_df.groupby("user_id")["neg_bal"].cumsum()

#     quovo_df["is_checking_or_savings"] = quovo_df["account_type"].isin(
#         ["Checking", "Savings"]
#     )
#     quovo_df.loc[
#         quovo_df["is_checking_or_savings"], "available_balance_if_checking_or_savings"
#     ] = quovo_df.loc[quovo_df["is_checking_or_savings"], "available_bal"]
#     quovo_df["quovo_largest_checking_or_savings_balance"] = quovo_df.groupby("user_id")[
#         "available_balance_if_checking_or_savings"
#     ].cummax()
#     quovo_df["quovo_available_bal"] = quovo_df["available_bal"]

#     quovo_df = utils.add_prefix(
#         quovo_df,
#         "quovo_",
#         exclusion=[
#             "user_id",
#             "borrower_id",
#             "business_account_number",
#             "transaction_datetime",
#         ],
#     )
#     return quovo_df
