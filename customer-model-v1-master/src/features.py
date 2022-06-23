"""
Building features from raw data.
"""

import pandas as pd
import numpy as np


def transform(df):
    """
    Build features from joined data.
    """
    #     df = pd.merge(df,
    #                   df.groupby('business_account_number')['transaction_datetime'].min()\
    #                   .rename('first_transaction_datetime').to_frame(),
    #                   how='left', on='business_account_number')

    #     df['lag_acc_open_first_transaction'] = (df['first_transaction_datetime'] - \
    #                                             df['date_account_opened']).dt.days

    #     df['days_since_first_deposit'] = (df['sample_date'] - \
    #                                       df['first_transaction_datetime']).dt.days

    #     df['age_money_account'] = (df['sample_date'] - \
    #                                df['date_account_opened']).dt.days

    #     df['days_since_last_transaction'] = (df['sample_date'] - df['transaction_datetime']).dt.days

    #     ### GIACT FEATURES
    #     df['giact_time_since_first_link'] = (df['sample_date'] - \
    #                                          df['giact_first_link_date']).dt.days
    #     df['giact_time_since_last_link'] = (df['sample_date'] - \
    #                                         df['giact_last_link_date']).dt.days

    return df
