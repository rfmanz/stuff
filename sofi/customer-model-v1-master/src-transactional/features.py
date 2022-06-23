"""
Building features from raw data.
"""

import pandas as pd
import numpy as np

from multiprocessing import Pool, cpu_count

def applyParallel(dfGrouped, func):
    """ Helper to parallelize apply over groupby """
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)

def func(df_):
    print("to overwrite")

def transform(transactions_df):
    """
    Build features from joined data.
    """
    global func
    
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

    ### TRANSACTION (not roll-ups) FEATURES
    transactions_df["transaction_as_pct_of_balance"] = transactions_df[
        "transaction_amount"
    ] / (transactions_df["real_ending_balance"] - transactions_df["transaction_amount"])

    transactions_df["last_transaction_datetime"] = transactions_df.groupby(
        "business_account_number"
    )["transaction_datetime"].shift(1)
    transactions_df["time_since_last_transaction"] = (
        transactions_df["transaction_datetime"]
        - transactions_df["last_transaction_datetime"]
    ).dt.days  # this relies on transactions we don't like not being included!

    ### TRANSACTION/ROLLUP FEATURES
    #     transactions_df['is_ach_return'] = (transactions_df['transaction_code'].isin(['DWACHRET', 'DDACHRET']) | \
    #                                         ((transactions_df['transaction_code'] == 'ACHDD') & \
    #                                          (transactions_df['transaction_amount'] < 0)))
    #     transactions_df['nr_past_ach_returns'] = transactions_df.groupby('business_account_number')['is_ach_return'].cumsum()
    #     transactions_df['nr_ach_returns_3d'] = transactions_df.groupby('business_account_number') \
    #                                                           .rolling('3d', min_periods=1,
    #                                                                    on='transaction_datetime') \
    #                                                            ['is_ach_return'].sum().values
    #     transactions_df['nr_ach_returns_10d'] = transactions_df.groupby('business_account_number') \
    #                                                            .rolling('10d', min_periods=1,
    #                                                                     on='transaction_datetime') \
    #                                                             ['is_ach_return'].sum().values
    #     transactions_df['nr_ach_returns_30d'] = transactions_df.groupby('business_account_number') \
    #                                                            .rolling('30d', min_periods=1,
    #                                                                        on='transaction_datetime') \
    #                                                             ['is_ach_return'].sum().values

    #     transactions_df['is_check_return'] = (transactions_df['transaction_code'].isin(['DWCKCB']) | \
    #                                           ((transactions_df['transaction_code'] == 'DDCK') & \
    #                                            (transactions_df['transaction_amount'] < 0)))
    #     transactions_df['nr_past_check_returns'] = transactions_df.groupby('business_account_number')['is_check_return'].cumsum()
    #     transactions_df['nr_check_returns_3d'] = transactions_df.groupby('business_account_number') \
    #                                                           .rolling('3d', min_periods=1,
    #                                                                    on='transaction_datetime') \
    #                                                            ['is_check_return'].sum().values
    #     transactions_df['nr_check_returns_10d'] = transactions_df.groupby('business_account_number') \
    #                                                            .rolling('10d', min_periods=1,
    #                                                                     on='transaction_datetime') \
    #                                                             ['is_check_return'].sum().values
    #     transactions_df['nr_check_returns_30d'] = transactions_df.groupby('business_account_number') \
    #                                                            .rolling('30d', min_periods=1,
    #                                                                        on='transaction_datetime') \
    #                                                             ['is_check_return'].sum().values

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
    
    def func(df_): return df_.rolling('30d',min_periods=1,on='transaction_datetime')['is_return'].sum()
    transactions_df['nr_returns_30d']=applyParallel(transactions_df.groupby('business_account_number'), func).values


    transactions_df["is_trans"] = transactions_df["transaction_code"].isin(
        deposit_transaction_codes + withdrawal_transaction_codes
    )

    transactions_df["is_deposit"] = transactions_df["transaction_code"].isin(
        deposit_transaction_codes
    ) & (transactions_df["transaction_amount"] > 0)
    transactions_df["nr_past_deposits"] = transactions_df.groupby(
        "business_account_number"
    )["is_deposit"].cumsum()

    def func(df_): return df_.rolling('3d',min_periods=1,on='transaction_datetime')['is_deposit'].sum()
    transactions_df['nr_deposits_3d']=applyParallel(transactions_df.groupby('business_account_number'), func).values
    
    #     transactions_df['nr_deposits_10d'] = transactions_df.groupby('business_account_number') \
    #                                                         .rolling('10d', min_periods=1,
    #                                                                  on='transaction_datetime') \
    #                                                         ['is_deposit'].sum().values
    def func(df_): return df_.rolling('30d',min_periods=1,on='transaction_datetime')['is_deposit'].sum()
    transactions_df['nr_deposits_30d']=applyParallel(transactions_df.groupby('business_account_number'), func).values


    transactions_df["nr_past_transactions"] = transactions_df.groupby(
        "business_account_number"
    )["business_account_number"].cumcount()
    
    def func(df_): return df_.rolling('3d',min_periods=1,on='transaction_datetime')['is_trans'].sum()
    transactions_df['nr_transactions_3d']=applyParallel(transactions_df.groupby('business_account_number'), func).values

    def func(df_): return df_.rolling('30d',min_periods=1,on='transaction_datetime')['is_trans'].sum()
    transactions_df['nr_transactions_30d']=applyParallel(transactions_df.groupby('business_account_number'), func).values

    transactions_df["pct_returned_deposits"] = (
        transactions_df["nr_past_returns"] / transactions_df["nr_past_deposits"]
    )
    transactions_df["pct_returned_deposits_30d"] = (
        transactions_df["nr_returns_30d"] / transactions_df["nr_deposits_30d"]
    )

    transactions_df["tamt_adjusted"] = transactions_df["transaction_amount"] * np.where(
        transactions_df["transaction_code"] == "ACHDD", -1, 1
    )
    transactions_df["return_dollar_amount"] = (
        transactions_df["is_return"] * transactions_df["tamt_adjusted"]
    )
    transactions_df["dollar_val_returns"] = transactions_df.groupby(
        "business_account_number"
    )["return_dollar_amount"].cumsum()
    
    def func(df_): return df_.rolling('3d',min_periods=1,on='transaction_datetime')['return_dollar_amount'].sum()
    transactions_df['dollar_val_returns_3d']=applyParallel(transactions_df.groupby('business_account_number'), func).values

    #     transactions_df['dollar_val_returns_10d'] = transactions_df.groupby('business_account_number') \
    #                                                                .rolling('10d', min_periods=1,
    #                                                                         on='transaction_datetime') \
    #                                                                ['return_dollar_amount'].sum().values
    def func(df_): return df_.rolling('30d',min_periods=1,on='transaction_datetime')['return_dollar_amount'].sum()
    transactions_df['dollar_val_returns_30d']=applyParallel(transactions_df.groupby('business_account_number'), func).values


    def func(df_): return df_.rolling('3d',min_periods=1,on='transaction_datetime')['real_ending_balance'].mean()
    transactions_df['mean_account_balance_3d']=applyParallel(transactions_df.groupby('business_account_number'), func).values

    def func(df_): return df_.rolling('30d',min_periods=1,on='transaction_datetime')['real_ending_balance'].mean()
    transactions_df['mean_account_balance_30d']=applyParallel(transactions_df.groupby('business_account_number'), func).values


    transactions_df["deposit_transaction_amount"] = (
        transactions_df["is_deposit"] * transactions_df["transaction_amount"]
    ).replace(np.nan, 0)
    
    def func(df_): return df_.rolling('3d',min_periods=1,on='transaction_datetime')['deposit_transaction_amount'].sum()
    transactions_df['sum_deposits_3d']=applyParallel(transactions_df.groupby('business_account_number'), func).values

    def func(df_): return df_.rolling('10d',min_periods=1,on='transaction_datetime')['deposit_transaction_amount'].sum()
    transactions_df['sum_deposits_10d']=applyParallel(transactions_df.groupby('business_account_number'), func).values

    def func(df_): return df_.rolling('30d',min_periods=1,on='transaction_datetime')['deposit_transaction_amount'].sum()
    transactions_df['sum_deposits_30d']=applyParallel(transactions_df.groupby('business_account_number'), func).values


    transactions_df["is_withdrawal"] = transactions_df["transaction_code"].isin(
        withdrawal_transaction_codes
    ) & (transactions_df["transaction_amount"] < 0)
    transactions_df["withdrawal_transaction_amount"] = (
        transactions_df["is_withdrawal"] * transactions_df["transaction_amount"]
    ).replace(np.nan, 0)
    
    
    def func(df_): return df_.rolling('3d',min_periods=1,on='transaction_datetime')['withdrawal_transaction_amount'].sum()
    transactions_df['sum_withdrawals_3d']=applyParallel(transactions_df.groupby('business_account_number'), func).values

    def func(df_): return df_.rolling('10d',min_periods=1,on='transaction_datetime')['withdrawal_transaction_amount'].sum()
    transactions_df['sum_withdrawals_10d']=applyParallel(transactions_df.groupby('business_account_number'), func).values

    def func(df_): return df_.rolling('30d',min_periods=1,on='transaction_datetime')['withdrawal_transaction_amount'].sum()
    transactions_df['sum_withdrawals_30d']=applyParallel(transactions_df.groupby('business_account_number'), func).values

    def func(df_): return df_.rolling('10d',min_periods=1,on='transaction_datetime')['deposit_transaction_amount'].mean()
    transactions_df['mean_deposits_10d']=applyParallel(transactions_df.groupby('business_account_number'), func).values

    def func(df_): return df_.expanding().mean()
    transactions_df['mean_deposits']=applyParallel(transactions_df.groupby('business_account_number')['deposit_transaction_amount'], func).values

    def func(df_): return df_.rolling('10d',min_periods=1,on='transaction_datetime')['withdrawal_transaction_amount'].mean()
    transactions_df['mean_withdrawals_10d']=applyParallel(transactions_df.groupby('business_account_number'), func).values

    def func(df_): return df_.expanding().mean()
    transactions_df['mean_withdrawals']=applyParallel(transactions_df.groupby('business_account_number')['withdrawal_transaction_amount'], func).values

    def func(df_): return df_.rolling('3d',min_periods=1,on='transaction_datetime')['deposit_transaction_amount'].max()
    transactions_df['max_deposits_3d']=applyParallel(transactions_df.groupby('business_account_number'), func).values

    def func(df_): return df_.rolling('30d',min_periods=1,on='transaction_datetime')['deposit_transaction_amount'].max()
    transactions_df['max_deposits_30d']=applyParallel(transactions_df.groupby('business_account_number'), func).values

    def func(df_): return df_.rolling('3d',min_periods=1,on='transaction_datetime')['withdrawal_transaction_amount'].min()
    transactions_df['max_withdrawals_3d']=applyParallel(transactions_df.groupby('business_account_number'), func).values

    def func(df_): return df_.rolling('30d',min_periods=1,on='transaction_datetime')['withdrawal_transaction_amount'].min()
    transactions_df['max_withdrawals_30d']=applyParallel(transactions_df.groupby('business_account_number'), func).values


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
    transactions_df["withdrawals_ratio"] = (
        transactions_df["sum_withdrawals_3d"] / transactions_df["sum_withdrawals_30d"]
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
