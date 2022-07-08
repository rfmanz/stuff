import numpy as np
import pandas as pd
from src.feature.utils import applyParallel

def transform(df):
    
    df = transform_(df)
    return df, {}

def func(df):
    raise NotImplemented("Use as a place holder for func in applyParallel")

def transform_(df):    
#     vantage_score
#     fico_score
#     total_outstanding_balance
#     credit_card_load_amount
#     bcc7120
#     plaid_max_avail_bal
#     plaid_min_avail_bal
#     plaid_days_since_first_link
#     quovo_available_bal
#     quovo_largest_checking_or_savings_balance
#     quovo_min_avail_bal
#     email_risk_score
#     fraud_score_2
#     phone_risk_score
#     fraud_score_1
#     address_risk_score
#     first_deposit_amount
#     sum_deposits_10d

    print("building ACH features")

    transactions_df = df
    
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
    
    

    transactions_df["trnx_is_return"] = transactions_df["trnx_transaction_code"].isin(
        ["DWCKCB", "DWACHRET", "DDACHRET"]
    ) | (
        (transactions_df["trnx_transaction_code"].isin(deposit_transaction_codes))
        & (transactions_df["trnx_transaction_amount"] < 0)
    )
    transactions_df["trnx_is_trans"] = transactions_df["trnx_transaction_code"].isin(
        deposit_transaction_codes + withdrawal_transaction_codes
    )

    transactions_df["trnx_is_deposit"] = transactions_df["trnx_transaction_code"].isin(
        deposit_transaction_codes
    ) & (transactions_df["trnx_transaction_amount"] > 0)

    
    transactions_df["trnx_deposit_transaction_amount"] = (
        transactions_df["trnx_is_deposit"] * transactions_df["trnx_transaction_amount"]
    ).replace(np.nan, 0)

    transactions_df["trnx_transaction_as_pct_of_balance"] = transactions_df[
        "trnx_transaction_amount"
    ] / (transactions_df["trnx_real_ending_balance"] - transactions_df["trnx_transaction_amount"])

    
    global func
    def func(df_): return df_.rolling('10d',min_periods=1,on='transaction_datetime')['trnx_deposit_transaction_amount'].sum()
    transactions_df['sum_deposits_10d']=applyParallel(transactions_df.groupby('business_account_number'), func).values


#     rolling_trns_as_pct_of_bal

# rolling_trns_as_pct_of_bal
    transactions_df['trnx_transaction_as_pct_of_balance_abs'] = transactions_df['trnx_transaction_as_pct_of_balance'].abs()

    def func(df_): return df_.rolling('7d',min_periods=1,on='transaction_datetime')['trnx_transaction_as_pct_of_balance_abs'].mean()
    transactions_df['trnx_rolling_trns_as_pct_of_bal']=applyParallel(transactions_df.groupby('business_account_number'), func).values


#     deposits_ratio

    def func(df_): return df_.rolling('3d',min_periods=1,on='transaction_datetime')['trnx_deposit_transaction_amount'].sum()
    transactions_df['trnx_sum_deposits_3d']=applyParallel(transactions_df.groupby('business_account_number'), func).values

    def func(df_): return df_.rolling('30d',min_periods=1,on='transaction_datetime')['trnx_deposit_transaction_amount'].sum()
    transactions_df['trnx_sum_deposits_30d']=applyParallel(transactions_df.groupby('business_account_number'), func).values

    transactions_df['trnx_deposits_ratio'] = transactions_df['trnx_sum_deposits_3d'] / transactions_df['trnx_sum_deposits_30d']

#     nr_transactions_per_day

    transactions_df = pd.merge(transactions_df, 
                      transactions_df.groupby('business_account_number')['transaction_datetime'].min()\
                      .rename('trnx_first_transaction_datetime').to_frame(), 
                      how='left', on='business_account_number')
    transactions_df['trnx_days_since_first_deposit'] = (transactions_df['transaction_datetime'] - \
                                                   transactions_df['trnx_first_transaction_datetime']).dt.days
    transactions_df['trnx_nr_past_transactions'] = transactions_df.groupby('business_account_number')\
                                                ['business_account_number'].cumcount()
    transactions_df['trnx_nr_transactions_per_day'] = transactions_df['trnx_nr_past_transactions'] / \
                                                 transactions_df['trnx_days_since_first_deposit']

#     nr_transactions_30d_div_nr_past_transactions
    def func(df_):
        return df_.rolling("30d", min_periods=1, on="transaction_datetime")[
            "trnx_is_trans"
        ].sum()

    transactions_df["trnx_nr_transactions_30d"] = applyParallel(
        transactions_df.groupby("business_account_number"), func
    ).values


    transactions_df['trnx_nr_transactions_30d_div_nr_past_transactions'] = transactions_df['trnx_nr_transactions_30d'] / transactions_df['trnx_nr_past_transactions']
#     nr_past_deposits
    transactions_df['trnx_nr_past_deposits'] = transactions_df.groupby('business_account_number')['trnx_is_deposit'].cumsum()
    
    return transactions_df