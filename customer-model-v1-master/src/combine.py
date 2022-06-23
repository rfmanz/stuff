import pandas as pd
import numpy as np
import datetime
import gc


cols_raw = [
    "first_deposit_amount",
    "vantage_score",
    "bcc7120",
    "email_risk_score",
    "fraud_score_2",
    "name_email_correlation",
    "transaction_as_pct_of_balance",
    "mean_account_balance_30d",
    "giact_time_since_last_link",
    "phone_risk_score",
    "name_address_correlation",
    "all8220",
    "lag_acc_open_first_transaction",
    "dollar_val_dd",
    "all7120",
    "sum_deposits_10d",
    "nr_past_transactions",
    "total_tradelines_open",
    "education_loan_amount",
    "address_risk_score",
    "iqt9415",
    "max_withdrawals_30d",
    "iln5520",
    "max_deposits_30d",
    "pct_returned_deposits",
    "giact_nr_decline",
    "nr_direct_deposits",
    "time_since_last_transaction",
    "bal_ratio",
    "name_phone_correlation",
    "giact_nr_other",
    "dollar_val_returns",
    "nr_trans_ratio",
    "iqt9413",
    "dollar_val_returns_3d",
    "nr_returns_30d",
    "credit_card_loan_amount",
    "fraud_score_1",
    "age_money_account",
]

cols_metadata = [
    "business_account_number",
    "borrower_id",
    "sample_date",
    "target",
    "transaction_code",
    "indeterminate"
]
# columns required

def combine_data(
    dynamic_full,
    static_full,
    date_sample_start,
    date_sample_end,
    all_features=False,
    filtering=True,
):

    ####################################
    #             dynamic
    ####################################

    # get dynamic sample range
    print("New Method!")
    print(f"dynamic sampling on dates from {date_sample_start} to {date_sample_end}")
    dynamic_full = dynamic_full[
        dynamic_full["transaction_datetime"].between(
            pd.to_datetime(date_sample_start), pd.to_datetime(date_sample_end)
        )
    ]  # keep transactions between sample date range

    dynamic_full = dynamic_full[
        ~(pd.to_datetime(dynamic_full["dtc"]) < dynamic_full["transaction_datetime"])
    ]  # remove transactions before account closed
    dynamic_full = dynamic_full[
        ~(
            pd.to_datetime(dynamic_full["chg_wrt_off_date"])
            < dynamic_full["transaction_datetime"]
        )
    ]  # remove transactions after the account is charged off

    dynamic_full["sample_date"] = dynamic_full["transaction_datetime"]

    # build indeterminate here? - move this to label next
#     dynamic_full["indeterminate"] = get_indeterminate_dynamic(dynamic_full)

    # shuffle the dataframe
    dynamic_full = dynamic_full.sample(frac=1, random_state=42).reset_index(drop=True)
    gc.collect()

    # select first 15 transactions for each customer (transactions in random order)
    dynamic_full = dynamic_full.groupby("borrower_id").head(15)
    gc.collect()

    ####################################
    #             Static
    ####################################

    static_full = static_full[
        ~(pd.to_datetime(static_full["dtc"]) <= static_full["sample_date"])
    ]
    static_full = static_full[
        ~(static_full["chg_wrt_off_date"] <= static_full["sample_date"])
    ]
    static_full["age_money_account"] = (
        static_full["sample_date"] - static_full["date_account_opened"]
    ).dt.days
    
    # build indeterminate here? - move this to label next
#     static_full["indeterminate"] = get_indeterminate_static(static_full)

    
    ####################################
    #          Put together
    ####################################

    static_full["is_static"] = True
    dynamic_full["is_static"] = False

    if ("nr_transactions_next_60d" in static_full.columns) and (
        "nr_transactions_next_60d" not in dynamic_full.columns
    ):
        # to keep static sample's col
        dynamic_full["nr_transactions_next_60d"] = np.nan

    if all_features:
        cols = dynamic_full.columns.intersection(static_full.columns)
    else:
        cols = cols_metadata + cols_raw

    modeling_df = pd.concat(
        [dynamic_full[cols], static_full[cols]], axis=0, ignore_index=True
    )

    if filtering:
        return modeling_df[~modeling_df.indeterminate]
    return modeling_df


# def get_indeterminate_dynamic(df):
#     """
#     indeterminate definitions

#     1. bad but recovered account balance
#     2. good but charged off
#     3. good but recently down
#     4. good but closed by risk
#     5. good but restricted
#     6. in-active
#     """

#     ind = ((df["target"] & (df["latest_acc_bal"] > 0)) 
#            | (~df["target"] & (  # 1
#         (~df["chg_wrt_off_date"].isna())  # 2
#         | (df["latest_acc_bal"] < 0)  # 3
#         | (
#             df["closed_reason"].isin(  # 4
#                 [
#                     "Closed by SoFi - Risk Request",
#                     "Closed by SoFi - Charge-Off / Write-Off",
#                 ]
#             )
#         )
#         | (df["restricted_reason"].str.startswith("No")))  # 5
#     ))
#     return ind


# def get_indeterminate_static(df):
#     """
#     indeterminate definitions

#     1. bad but recovered account balance
#     2. good but charged off
#     3. good but recently down
#     4. good but closed by risk
#     5. good but restricted
#     6. in-active
#     """

#     ind = (
#         (df["nr_transactions_next_60d"] == 0)  # 6
#         | (df["target"] & (df["latest_acc_bal"] > 0))  # 1
#         | (
#             ~df["target"]
#             & (
#                 ~df["chg_wrt_off_date"].isna()  # 2
#                 | (df["latest_acc_bal"] < 0)  # 3
#                 | df["restricted_reason"].str.startswith("No")  # 5
#                 | df["closed_reason"].isin(  # 4
#                     [
#                         "Closed by SoFi - Risk Request",
#                         "Closed by SoFi - Charge-Off / Write-Off",
#                     ]
#                 )
#             )
#         )
#     )

#     return ind


