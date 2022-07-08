"""
Building features from raw data.
"""

import pandas as pd


def transform(transactions_df):
    """
    Build features from processed DW data.
    """
    # nr_direct_deposits
    transactions_df['is_dd'] = transactions_df['transaction_code'].isin(['ACHINDD'])
    transactions_df['nr_direct_deposits'] = transactions_df.groupby('borrower_id')['is_dd'].cumsum()

    # nr_past_returns
    transactions_df['is_return'] = transactions_df['transaction_code'].isin(['DWACHRET', 'DWCKCB']) | \
                               ((transactions_df['transaction_code'] == 'ACHDD') &\
                                (transactions_df['transaction_amount'] < 0))
    transactions_df['nr_past_returns'] = transactions_df.groupby('borrower_id')['is_return'].cumsum()
    
    # days_since_first_transactions
    transactions_df = pd.merge(transactions_df, 
                               transactions_df.groupby('borrower_id') \
                               ['transaction_datetime'].min().rename('first_transaction_datetime') \
                               .to_frame(), on='borrower_id', how='left')
    transactions_df['days_since_first_transaction'] = (transactions_df['transaction_datetime'] - \
                                                       transactions_df['first_transaction_datetime']).dt.days

    # nr_transactions_per_day
    transactions_df['nr_past_transactions'] = transactions_df.groupby('borrower_id')['borrower_id'].cumcount()
    transactions_df['nr_transactions_per_day'] = transactions_df['nr_past_transactions'] / \
                                                 transactions_df['days_since_first_transaction']
    
    # transaction_as_pct_of_balance
    transactions_df['transaction_as_pct_of_balance'] = transactions_df['transaction_amount'] / \
                                                       (transactions_df['account_ending_balance'] - \
                                                        transactions_df['transaction_amount'])

    # rolling_trns_as_pct_of_bal
    transactions_df['transaction_as_pct_of_balance_abs'] = transactions_df['transaction_as_pct_of_balance'].abs()
    transactions_df['rolling_trns_as_pct_of_bal'] = transactions_df.groupby('borrower_id') \
                                                                   .rolling('7d', min_periods=1, 
                                                                            on='transaction_datetime') \
                                                    ['transaction_as_pct_of_balance_abs'].mean().values

    # transaction_as_pct_of_bal_min
    transactions_df['transaction_as_pct_of_bal_min'] = transactions_df.groupby('borrower_id') \
                                                                      .rolling('7d', min_periods=1, 
                                                                               on='transaction_datetime') \
                                                       ['transaction_as_pct_of_balance'].min().values

    # rolling_mean_acc_bal <<< THIS IS WRONG? something going wrong!!
    transactions_df['rolling_mean_acc_bal'] = transactions_df.groupby('borrower_id') \
                                                             .rolling('14d', min_periods=1, 
                                                                      on='transaction_datetime')\
                                              ['account_ending_balance'].mean().values
    return transactions_df
