"""
Building features from raw data.
"""
import gc

import pandas as pd
import numpy as np

from tqdm import tqdm


def transform(transactions_df):
    """
    Build features from joined data.
    """
    # Sort data by transaction datetime.
    transactions_df = transactions_df.sort_values(
        by=["business_account_number", "transaction_datetime"]
    )

    ### BANKING FEATURES
    transactions_df = pd.merge(
        transactions_df,
        transactions_df.groupby("business_account_number")["transaction_datetime"]
        .min()
        .rename("first_transaction_datetime")
        .to_frame(),
        how="left",
        on="business_account_number",
    )
    transactions_df["days_since_first_deposit"] = (
        transactions_df["transaction_datetime"]
        - transactions_df["first_transaction_datetime"]
    ).dt.days

    transactions_df["age_money_account"] = (
        transactions_df["transaction_datetime"] - transactions_df["odt"]
    ).dt.days

    transactions_df["lag_acc_open_first_transaction"] = (
        transactions_df["first_transaction_datetime"] - transactions_df["odt"]
    ).dt.days

    transactions_df["first_deposit_amount"] = transactions_df[
        "afdep"
    ]  # TODO: this is a waste of space.

    ### CREDIT FEATURES
    # No transformations on credit attributes at this time.
    # Eventually will transform because of default values.

    ### GIACT FEATURES
    transactions_df["giact_time_since_first_link"] = (
        transactions_df["transaction_datetime"]
        - transactions_df["giact_first_link_date"]
    ).dt.days
    transactions_df["giact_time_since_last_link"] = (
        transactions_df["transaction_datetime"]
        - transactions_df["giact_last_link_date"]
    ).dt.days

    ### TMX FEATURES
    # Nothing at the moment.

    ### External bank linkages:
    res = []
    curr = None
    counter = {}

    for row in tqdm(
        transactions_df[
            [
                "business_account_number",
                "transaction_datetime",
                "external_account_number",
                "external_institution_id",
                "transaction_amount",
            ]
        ].values
    ):
        if row[0] != curr:
            curr = row[0]
            counter = {}

        if not row[2]:
            res.append([None for i in range(6)])
            continue

        out = []

        external_account_number = row[2]

        if external_account_number not in counter:
            counter[external_account_number] = {}

        # nr past transactions with this account
        if "nr_trans_with_acc" in counter[external_account_number]:
            counter[external_account_number]["nr_trans_with_acc"] += 1
        else:
            counter[external_account_number]["nr_trans_with_acc"] = 1
        out.append(counter[external_account_number]["nr_trans_with_acc"] - 1)

        # first transaction dt
        if "first_transaction_dt" not in counter[external_account_number]:
            counter[external_account_number]["first_transaction_dt"] = row[1]
        out.append(counter[external_account_number]["first_transaction_dt"])

        # last transaction_dt
        if "last_transaction_dt" not in counter[external_account_number]:
            counter[external_account_number]["last_transaction_dt"] = None
        out.append(counter[external_account_number]["last_transaction_dt"])
        counter[external_account_number]["last_transaction_dt"] = row[1]

        # sum pos/neg transactions with acct
        if "sum_pos_trans" not in counter[external_account_number]:
            counter[external_account_number]["sum_pos_trans"] = 0
        if "sum_neg_trans" not in counter[external_account_number]:
            counter[external_account_number]["sum_neg_trans"] = 0
        out.append(counter[external_account_number]["sum_pos_trans"])
        out.append(counter[external_account_number]["sum_neg_trans"])

        if row[4] >= 0:
            counter[external_account_number]["sum_pos_trans"] += row[4]
        else:
            counter[external_account_number]["sum_neg_trans"] += row[4]

        if "rolling_mean_pos_trans" not in counter[external_account_number]:
            counter[external_account_number]["rolling_mean_pos_trans"] = row[4]
            out.append(None)
        else:
            out.append(counter[external_account_number]["rolling_mean_pos_trans"])
            counter[external_account_number]["rolling_mean_pos_trans"] = (
                counter[external_account_number]["rolling_mean_pos_trans"] + row[4]
            ) / 2

        res.append(out)

    ea_cols = [
        "nr_trans_with_acc",
        "first_trans_with_ea_dt",
        "last_trans_with_ea_dt",
        "sum_pos_trans_ea",
        "sum_neg_trans_ea",
        "rolling_mean_pos_trans_ea",
    ]
    transactions_df = transactions_df.assign(**dict.fromkeys(ea_cols, np.nan))
    transactions_df[ea_cols] = res

    del res

    # ea -> external account
    transactions_df["time_since_first_trans_ea"] = (
        transactions_df["transaction_datetime"]
        - transactions_df["first_trans_with_ea_dt"]
    ).dt.days
    transactions_df["time_since_last_trans_ea"] = (
        transactions_df["transaction_datetime"]
        - transactions_df["last_trans_with_ea_dt"]
    ).dt.days

    transactions_df["ratio_all_ea_trans_div_tamt"] = (
        transactions_df["sum_pos_trans_ea"] / transactions_df["transaction_amount"]
    )
    transactions_df["ratio_rolling_mean_ea_tamt_div_tamt"] = (
        transactions_df["rolling_mean_pos_trans_ea"]
        / transactions_df["transaction_amount"]
    )

    ### TRANSACTION (not roll-ups) FEATURES
    transactions_df["transaction_as_pct_of_balance"] = transactions_df[
        "transaction_amount"
    ] / (transactions_df["real_ending_balance"] - transactions_df["transaction_amount"])

    transactions_df["last_transaction_datetime"] = transactions_df.groupby(
        "business_account_number"
    )["transaction_datetime"].shift(1)
    transactions_df["last_transaction_code"] = transactions_df.groupby(
        "business_account_number"
    )["transaction_code"].shift(1)

    transactions_df["time_since_last_transaction"] = (
        transactions_df["transaction_datetime"]
        - transactions_df["last_transaction_datetime"]
    ).dt.days  # this relies on transactions we don't like not being included!

    deposit_transaction_codes = [
        "POSDD",
        "ACHDD",
        "ACHDDIN",
        "ACHINDD",
        "DDCK",
        "DDMBR",
        "DD",
    ]
    withdrawal_transaction_codes = [
        "POSDW",
        "ACHDW",
        "ACHDWIN",
        "DWATM",
        "DWATMI",
        "DWCK",
        "DWBILLPAY",
        "DWCRDBILLPAY",
        "DWMBR",
        "ACHDWP2P",
        "DWWIRE",
        "DBDWWIRE",
        "DWTRF",
        "DBDW",
        "DWSLROTP",
        "DW",
    ]

    transactions_df["is_return"] = transactions_df["transaction_code"].isin(
        ["DWCKCB", "DWACHRET", "DDACHRET"]
    ) | (
        (transactions_df["transaction_code"].isin(deposit_transaction_codes))
        & (transactions_df["transaction_amount"] < 0)
    )
    transactions_df["nr_past_returns"] = transactions_df.groupby(
        "business_account_number"
    )["is_return"].cumsum()
    transactions_df["nr_returns_30d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("30d", min_periods=1, on="transaction_datetime")["is_return"]
        .sum()
        .values
    )

    transactions_df["is_trans"] = transactions_df["transaction_code"].isin(
        deposit_transaction_codes + withdrawal_transaction_codes
    )

    transactions_df["is_deposit"] = transactions_df["transaction_code"].isin(
        deposit_transaction_codes
    ) & (transactions_df["transaction_amount"] > 0)
    transactions_df["nr_past_deposits"] = transactions_df.groupby(
        "business_account_number"
    )["is_deposit"].cumsum()

    transactions_df["nr_deposits_3d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("3d", min_periods=1, on="transaction_datetime")["is_deposit"]
        .sum()
        .values
    )
    transactions_df["nr_deposits_30d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("30d", min_periods=1, on="transaction_datetime")["is_deposit"]
        .sum()
        .values
    )

    transactions_df["nr_past_transactions"] = transactions_df.groupby(
        "business_account_number"
    )["business_account_number"].cumcount()
    transactions_df["nr_transactions_3d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("3d", min_periods=1, on="transaction_datetime")["is_trans"]
        .sum()
        .values
    )
    transactions_df["nr_transactions_30d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("30d", min_periods=1, on="transaction_datetime")["is_trans"]
        .sum()
        .values
    )

    transactions_df["pct_returned_deposits"] = (
        transactions_df["nr_past_returns"] / transactions_df["nr_past_deposits"]
    )
    transactions_df["pct_returned_deposits_30d"] = (
        transactions_df["nr_returns_30d"] / transactions_df["nr_deposits_30d"]
    )

    transactions_df["nr_transactions_30d_div_nr_past_transactions"] = (
        transactions_df["nr_transactions_30d"] / transactions_df["nr_past_transactions"]
    )

    transactions_df["tamt_adjusted"] = transactions_df["transaction_amount"] * np.where(
        transactions_df["transaction_code"] == "ACHDD", -1, 1
    )

    transactions_df["mean_account_balance_3d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("3d", min_periods=1, on="transaction_datetime")["real_ending_balance"]
        .mean()
        .values
    )

    transactions_df["mean_account_balance_30d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("30d", min_periods=1, on="transaction_datetime")["real_ending_balance"]
        .mean()
        .values
    )

    transactions_df["std_account_balance_3d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("3d", min_periods=1, on="transaction_datetime")["real_ending_balance"]
        .std()
        .values
    )
    transactions_df["std_account_balance_30d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("30d", min_periods=1, on="transaction_datetime")["real_ending_balance"]
        .std()
        .values
    )

    transactions_df["deposit_transaction_amount"] = (
        transactions_df["is_deposit"] * transactions_df["transaction_amount"]
    ).replace(np.nan, 0)
    transactions_df["sum_deposits_3d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("3d", min_periods=1, on="transaction_datetime")[
            "deposit_transaction_amount"
        ]
        .sum()
        .values
    )
    transactions_df["sum_deposits_10d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("10d", min_periods=1, on="transaction_datetime")[
            "deposit_transaction_amount"
        ]
        .sum()
        .values
    )
    transactions_df["sum_deposits_30d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("30d", min_periods=1, on="transaction_datetime")[
            "deposit_transaction_amount"
        ]
        .sum()
        .values
    )

    transactions_df["is_withdrawal"] = transactions_df["transaction_code"].isin(
        withdrawal_transaction_codes
    ) & (transactions_df["transaction_amount"] < 0)
    transactions_df["withdrawal_transaction_amount"] = (
        transactions_df["is_withdrawal"] * transactions_df["transaction_amount"]
    ).replace(np.nan, 0)
    transactions_df["sum_withdrawals_3d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("3d", min_periods=1, on="transaction_datetime")[
            "withdrawal_transaction_amount"
        ]
        .sum()
        .values
    )
    transactions_df["sum_withdrawals_10d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("10d", min_periods=1, on="transaction_datetime")[
            "withdrawal_transaction_amount"
        ]
        .sum()
        .values
    )
    transactions_df["sum_withdrawals_30d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("30d", min_periods=1, on="transaction_datetime")[
            "withdrawal_transaction_amount"
        ]
        .sum()
        .values
    )

    transactions_df["mean_deposits_10d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("10d", min_periods=1, on="transaction_datetime")[
            "deposit_transaction_amount"
        ]
        .mean()
        .values
    )
    transactions_df["mean_deposits"] = (
        transactions_df.groupby("business_account_number")["deposit_transaction_amount"]
        .expanding()
        .mean()
        .values
    )

    transactions_df["mean_deposits_10d_div_mean_deposits"] = (
        transactions_df["mean_deposits_10d"] / transactions_df["mean_deposits"]
    )

    transactions_df["mean_withdrawals_10d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("10d", min_periods=1, on="transaction_datetime")[
            "withdrawal_transaction_amount"
        ]
        .mean()
        .values
    )
    transactions_df["mean_withdrawals"] = (
        transactions_df.groupby("business_account_number")[
            "withdrawal_transaction_amount"
        ]
        .expanding()
        .mean()
        .values
    )
    transactions_df["mean_withdrawals_10d_div_mean_withdrawals"] = (
        transactions_df["mean_withdrawals_10d"] / transactions_df["mean_withdrawals"]
    )

    transactions_df["max_deposits"] = (
        transactions_df.groupby("business_account_number")["deposit_transaction_amount"]
        .expanding()
        .max()
        .values
    )
    transactions_df["max_deposits_3d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("3d", min_periods=1, on="transaction_datetime")[
            "deposit_transaction_amount"
        ]
        .max()
        .values
    )
    transactions_df["max_deposits_10d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("10d", min_periods=1, on="transaction_datetime")[
            "deposit_transaction_amount"
        ]
        .max()
        .values
    )
    transactions_df["max_deposits_10d_div_mean_deposits"] = (
        transactions_df["max_deposits_10d"] / transactions_df["mean_deposits"]
    )
    transactions_df["max_deposits_10d_div_mean_account_balance_30d"] = (
        transactions_df["max_deposits_10d"]
        / transactions_df["mean_account_balance_30d"]
    )

    transactions_df["max_withdrawals"] = (
        transactions_df.groupby("business_account_number")[
            "withdrawal_transaction_amount"
        ]
        .expanding()
        .max()
        .values
    )
    transactions_df["max_withdrawals_3d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("3d", min_periods=1, on="transaction_datetime")[
            "withdrawal_transaction_amount"
        ]
        .min()
        .values
    )
    transactions_df["max_withdrawals_10d"] = (
        transactions_df.groupby("business_account_number")
        .rolling("10d", min_periods=1, on="transaction_datetime")[
            "withdrawal_transaction_amount"
        ]
        .min()
        .values
    )
    transactions_df["max_withdrawals_10d_div_mean_withdrawals"] = (
        transactions_df["max_withdrawals_10d"] / transactions_df["mean_withdrawals"]
    )

    transactions_df["nr_trans_ratio"] = (
        transactions_df["nr_transactions_3d"] / transactions_df["nr_transactions_30d"]
    )
    transactions_df["bal_ratio"] = (
        transactions_df["mean_account_balance_3d"]
        / transactions_df["mean_account_balance_30d"]
    )
    transactions_df["deposits_ratio"] = (
        transactions_df["sum_deposits_3d"] / transactions_df["sum_deposits_30d"]
    )

    transactions_df["is_dd"] = transactions_df["transaction_code"] == "ACHINDD"
    transactions_df["nr_direct_deposits"] = transactions_df.groupby(
        "business_account_number"
    )["is_dd"].cumsum()
    transactions_df["dd_dollar_amount"] = (
        transactions_df["is_dd"] * transactions_df["transaction_amount"]
    )
    transactions_df["dollar_val_dd"] = transactions_df.groupby(
        "business_account_number"
    )["dd_dollar_amount"].cumsum()

    ################################################
    #      DEPOSIT V1 features for bencharking:
    ################################################
    transactions_df["rolling_mean_acc_bal"] = (
        transactions_df.groupby("business_account_number")
        .rolling("14d", min_periods=1, on="transaction_datetime")["real_ending_balance"]
        .mean()
        .values
    )

    # nr_transactions_per_day
    transactions_df["nr_past_transactions"] = transactions_df.groupby(
        "business_account_number"
    )["business_account_number"].cumcount()
    transactions_df["nr_transactions_per_day"] = (
        transactions_df["nr_past_transactions"]
        / transactions_df["days_since_first_deposit"]
    )

    # transaction_as_pct_of_balance
    transactions_df["transaction_as_pct_of_balance"] = transactions_df[
        "transaction_amount"
    ] / (transactions_df["real_ending_balance"] - transactions_df["transaction_amount"])

    # rolling_trns_as_pct_of_bal
    transactions_df["transaction_as_pct_of_balance_abs"] = transactions_df[
        "transaction_as_pct_of_balance"
    ].abs()
    transactions_df["rolling_trns_as_pct_of_bal"] = (
        transactions_df.groupby("business_account_number")
        .rolling("7d", min_periods=1, on="transaction_datetime")[
            "transaction_as_pct_of_balance_abs"
        ]
        .mean()
        .values
    )

    # transaction_as_pct_of_bal_min
    transactions_df["transaction_as_pct_of_bal_min"] = (
        transactions_df.groupby("business_account_number")
        .rolling("7d", min_periods=1, on="transaction_datetime")[
            "transaction_as_pct_of_balance"
        ]
        .min()
        .values
    )

    # rolling_mean_acc_bal <<< THIS IS WRONG? something going wrong!!
    transactions_df["rolling_mean_acc_bal"] = (
        transactions_df.groupby("business_account_number")
        .rolling("14d", min_periods=1, on="transaction_datetime")["real_ending_balance"]
        .mean()
        .values
    )

    ######################################################
    #    FEATURES FOR DEBUGGING - has data snooping bias
    ######################################################
    transactions_df = pd.merge(
        transactions_df,
        transactions_df.groupby("business_account_number")["business_account_number"]
        .count()
        .rename("nr_transactions_all_time")
        .reset_index(),
        how="inner",
        on="business_account_number",
    )

    transactions_df = pd.merge(
        transactions_df,
        transactions_df[transactions_df["is_return"]]
        .groupby("business_account_number")["transaction_datetime"]
        .min()
        .rename("first_return_date")
        .reset_index(),
        how="left",
        on="business_account_number",
    )

    transactions_df = pd.merge(
        transactions_df,
        transactions_df[
            transactions_df["transaction_code"].isin(
                ["DDCHGOFF", "DDWRTOFF", "DDFRDWO"]
            )
        ]
        .groupby("business_account_number")["transaction_datetime"]
        .min()
        .rename("chg_wrt_off_date")
        .reset_index(),
        how="left",
        on="business_account_number",
    )

    transactions_df = pd.merge(
        transactions_df,
        transactions_df.groupby("business_account_number")["is_return"]
        .sum()
        .rename("nr_returns_all_time")
        .reset_index(),
        how="inner",
        on="business_account_number",
    )

    ### ENCODINGS
    #     mdict = {'ACHDD': 0, 'ACHDDIN': 1, 'DD': 2, 'DDCK': 3, 'ACHINDD': 4}
    #     transactions_df['transaction_code_encoded'] = transactions_df['transaction_code'].map(mdict)

    return transactions_df
