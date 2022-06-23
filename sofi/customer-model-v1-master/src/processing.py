"""
Functions for processing raw data files.
"""

import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore")


def process_banking_accounts(banking_accounts_df, **kwargs):
    """
    Process banking data.
    """
    print("Processing banking data.")
    data_pull_date = kwargs["data_pull_date"]

    banking_accounts_df["date_account_opened"] = pd.to_datetime(
        banking_accounts_df["date_account_opened"]
    ).dt.tz_localize(None)
    banking_accounts_df["date_account_closed"] = pd.to_datetime(
        banking_accounts_df["date_account_closed"]
    ).dt.tz_localize(None)

    banking_accounts_df["account_age_days_asof_pull"] = (
        data_pull_date - banking_accounts_df["date_account_opened"]
    ).dt.days

    return banking_accounts_df


def process_experian_credit_pull(ecp_df, **kwargs):
    """
    Process experian credit pull data.
    """
    print("Processing experian credit pull data.")
    ecp_df["credit_pull_date"] = pd.to_datetime(ecp_df["credit_pull_date"])
    ecp_df_isna = ecp_df.credit_pull_date.isna()
    if ecp_df_isna.any():
        print(f"Missing values in ecp_df. Dropping {ecp_df_isna.sum()} na")
    ecp_df.dropna(subset=["credit_pull_date"], inplace=True)

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

    return giact_df


def process_quovo(quovo_df, **kwargs):
    """
    Process quovo data.
    """
    quovo_df["user_id"] = quovo_df["party_id"]
    quovo_df["available_balance"] = quovo_df["available_balance"].astype(float)
    quovo_df["fetched_from_quovo_dt"] = pd.to_datetime(
        quovo_df["fetched_from_quovo_dt"]
    ).dt.tz_localize(None)

    quovo_df = quovo_df.sort_values(by=["fetched_from_quovo_dt", "user_id"])

    quovo_df["pos_bal"] = quovo_df["available_balance"] * (
        quovo_df["available_balance"] > 0
    )
    quovo_df["neg_bal"] = quovo_df["available_balance"] * (
        quovo_df["available_balance"] < 0
    )

    quovo_df["quovo_first_link_date"] = quovo_df.groupby("user_id")[
        "fetched_from_quovo_dt"
    ].cummin()
    quovo_df["quovo_last_link_date"] = quovo_df.groupby("user_id")[
        "fetched_from_quovo_dt"
    ].cummax()

    quovo_df["quovo_max_avail_bal"] = quovo_df.groupby("user_id")[
        "available_balance"
    ].cummax()
    quovo_df["quovo_min_avail_bal"] = quovo_df.groupby("user_id")[
        "available_balance"
    ].cummin()

    quovo_df["quovo_total_pos_bal"] = quovo_df.groupby("user_id")["pos_bal"].cumsum()
    quovo_df["quovo_total_neg_bal"] = quovo_df.groupby("user_id")["neg_bal"].cumsum()

    quovo_df["quovo_max_nr_transactions"] = quovo_df.groupby("user_id")[
        "nr_transactions"
    ].cummax()
    quovo_df["quovo_min_nr_transactions"] = quovo_df.groupby("user_id")[
        "nr_transactions"
    ].cummin()
    quovo_df["quovo_total_nr_transactions"] = quovo_df.groupby("user_id")[
        "nr_transactions"
    ].cumsum()

    quovo_df["is_checking_or_savings"] = quovo_df["account_type"].isin(
        ["Checking", "Savings"]
    )
    quovo_df.loc[
        quovo_df["is_checking_or_savings"], "available_balance_if_checking_or_savings"
    ] = quovo_df.loc[quovo_df["is_checking_or_savings"], "available_balance"]
    quovo_df["quovo_largest_checking_or_savings_balance"] = quovo_df.groupby("user_id")[
        "available_balance_if_checking_or_savings"
    ].cummax()
    quovo_df["quovo_nr_linked_checking_or_savings"] = quovo_df.groupby("user_id")[
        "is_checking_or_savings"
    ].cumsum()

    return quovo_df


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

    return tmx_df


def process_transactions(transactions_df, **kwargs):
    """
    Process transactions data.

    good transactions:
    WHERE etc IN ('ACHDD',
              'ACHDDIN',
              'DD',
              'DDCK',
              'ACHINDD',
              'POSDW',
              'ACHDW',
              'ACHDWIN',
              'DWATM',
              'DWATMI',
              'DWCK',
              'DWBILLPAY',
              'DWCRDBILLPAY',
              'DWMBR',
              'ACHDWP2P',
              'DWWIRE',
              'DBDWWIRE',
              'DWTRF',
              'DBDW',
              'DWSLROTP',
              'DW')
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

    trans_df_dt_isna = transactions_df.transaction_datetime.isna()
    if trans_df_dt_isna.any():
        print(
            f"Missing values in transactions_df.transaction_datetime. Dropping {trans_df_dt_isna.sum()} na"
        )
    transactions_df.dropna(subset=["transaction_datetime"], inplace=True)

    #     transactions_df['user_id'] = transactions_df['zsofiid'].replace('', None).str.split(',', expand=True)[0].astype(int)
    #     transactions_df['joint_user_id'] = transactions_df['zsofiid'].str.split(',', expand=True)[1].astype(float)
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

    # Target def stuff
    #     transactions_df['is_chg_wrt_off'] = transactions_df['transaction_code'].isin(['DDCHGOFF', 'DDFRDWO', 'DDWRTOFF'])
    #     transactions_df = pd.merge(transactions_df,
    #                                transactions_df.groupby('business_account_number')['is_chg_wrt_off'].max().rename('ever_chg_wrt_off').reset_index(),
    #                                how='inner',
    #                                on='business_account_number')

    #     ## Make transactional roll-ups
    #     # Sort data by transaction datetime.
    #     transactions_df = transactions_df.sort_values(by=['business_account_number', 'transaction_datetime'])

    #     ### ROLL-UPS USED FOR TARGET DEFINTION.
    #     transactions_df['has_chg_wrt_off_in_past_120d'] = transactions_df.groupby('business_account_number') \
    #                                                                      .rolling('120d', min_periods=1,
    #                                                                               on='transaction_datetime') \
    #                                                                       ['is_chg_wrt_off'].max().values

    #     deposit_data_types = ['ACHDD', 'ACHDDIN', 'ACHINDD', 'DDCK', 'DD',
    #                           'DBDDWIRE', 'DBDD']
    #     deposit_return_types = ['DWCKCB', 'DWACHRET']

    #     # mark all returns as returns
    #     transactions_df['is_return'] = (transactions_df['transaction_code'].isin(deposit_data_types) & \
    #                                     (transactions_df['transaction_amount'] < 0)) | \
    #                                    (transactions_df['transaction_code'].isin(deposit_return_types))

    #     transactions_df['nr_returns_in_past_30d'] = transactions_df.groupby('business_account_number') \
    #                                                                .rolling('30d', min_periods=1,
    #                                                                          on='transaction_datetime') \
    #                                                                ['is_return'].sum().values

    #     ### TRANSACTION (not roll-ups) FEATURES
    transactions_df = pd.merge(
        transactions_df,
        transactions_df.groupby("business_account_number")["transaction_datetime"]
        .min()
        .rename("first_transaction_datetime")
        .to_frame(),
        how="left",
        on="business_account_number",
    )

    #     transactions_df['transaction_as_pct_of_balance'] = transactions_df['transaction_amount'] / \
    #                                                        (transactions_df['real_ending_balance'] - \
    #                                                         transactions_df['transaction_amount'])

    #     transactions_df['last_transaction_datetime'] = transactions_df.groupby('business_account_number')['transaction_datetime'].shift(1)
    #     transactions_df['time_since_last_transaction'] = (transactions_df['transaction_datetime'] - \
    #                                                       transactions_df['last_transaction_datetime']).dt.days # this relies on transactions we don't like not being included!

    #     ### TRANSACTION/ROLLUP FEATURES
    #     transactions_df['mean_trans_as_pct_of_bal'] = transactions_df.groupby('business_account_number') \
    #                                                                  .rolling('30d', min_periods=1,
    #                                                                           on='transaction_datetime') \
    #                                                                   ['transaction_as_pct_of_balance'].mean().values

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

    #     deposit_transaction_codes = ['POSDD', 'ACHDD', 'ACHDDIN', 'ACHINDD', 'DDCK', 'DDMBR', 'DD']
    #     withdrawal_transaction_codes = ['POSDW', 'ACHDW', 'ACHDWIN', 'DWATM', 'DWATMI', 'DWCK', 'DWBILLPAY',
    #                                     'DWCRDBILLPAY', 'DWMBR', 'ACHDWP2P', 'DWWIRE', 'DBDWWIRE', 'DWTRF', 'DBDW', 'DWSLROTP', 'DW']

    #     transactions_df['nr_past_returns'] = transactions_df.groupby('business_account_number')['is_return'].cumsum()
    #     transactions_df['nr_returns_30d'] = transactions_df.groupby('business_account_number') \
    #                                                        .rolling('30d', min_periods=1,
    #                                                                 on='transaction_datetime') \
    #                                                         ['is_ach_return'].sum().values

    #     transactions_df['is_trans'] = transactions_df['transaction_code'].isin(deposit_transaction_codes + withdrawal_transaction_codes)

    #     transactions_df['is_deposit'] = transactions_df['transaction_code'].isin(deposit_transaction_codes) & \
    #                                     (transactions_df['transaction_amount'] > 0)
    #     transactions_df['nr_past_deposits'] = transactions_df.groupby('business_account_number')['is_deposit'].cumsum()

    #     transactions_df['nr_deposits_3d'] = transactions_df.groupby('business_account_number') \
    #                                                        .rolling('3d', min_periods=1,
    #                                                                 on='transaction_datetime') \
    #                                                        ['is_deposit'].sum().values
    #     transactions_df['nr_deposits_10d'] = transactions_df.groupby('business_account_number') \
    #                                                         .rolling('10d', min_periods=1,
    #                                                                  on='transaction_datetime') \
    #                                                         ['is_deposit'].sum().values
    #     transactions_df['nr_deposits_30d'] = transactions_df.groupby('business_account_number') \
    #                                                         .rolling('30d', min_periods=1,
    #                                                                  on='transaction_datetime') \
    #                                                         ['is_deposit'].sum().values

    #     transactions_df['nr_past_transactions'] = transactions_df.groupby('business_account_number')['business_account_number'].cumcount()
    #     transactions_df['nr_transactions_3d'] = transactions_df.groupby('business_account_number') \
    #                                                        .rolling('3d', min_periods=1,
    #                                                                 on='transaction_datetime') \
    #                                                        ['is_trans'].sum().values
    #     transactions_df['nr_transactions_10d'] = transactions_df.groupby('business_account_number') \
    #                                                         .rolling('10d', min_periods=1,
    #                                                                  on='transaction_datetime') \
    #                                                         ['is_trans'].sum().values
    #     transactions_df['nr_transactions_30d'] = transactions_df.groupby('business_account_number') \
    #                                                         .rolling('30d', min_periods=1,
    #                                                                  on='transaction_datetime') \
    #                                                         ['is_trans'].sum().values

    #     transactions_df['pct_returned_deposits'] = transactions_df['nr_past_returns'] / \
    #                                                transactions_df['nr_past_deposits']
    #     transactions_df['pct_returned_deposits_30d'] = transactions_df['nr_returns_30d'] / \
    #                                                    transactions_df['nr_deposits_30d']

    #     transactions_df['tamt_adjusted'] = transactions_df['transaction_amount'] * np.where(transactions_df['transaction_code'] == 'ACHDD', -1, 1)
    #     transactions_df['return_dollar_amount'] = transactions_df['is_return'] * transactions_df['tamt_adjusted']
    #     transactions_df['dollar_val_returns'] = transactions_df.groupby('business_account_number')['return_dollar_amount'].cumsum()
    #     transactions_df['dollar_val_returns_3d'] = transactions_df.groupby('business_account_number') \
    #                                                               .rolling('3d', min_periods=1,
    #                                                                        on='transaction_datetime') \
    #                                                               ['return_dollar_amount'].sum().values
    #     transactions_df['dollar_val_returns_10d'] = transactions_df.groupby('business_account_number') \
    #                                                                .rolling('10d', min_periods=1,
    #                                                                         on='transaction_datetime') \
    #                                                                ['return_dollar_amount'].sum().values
    #     transactions_df['dollar_val_returns_30d'] = transactions_df.groupby('business_account_number') \
    #                                                                .rolling('30d', min_periods=1,
    #                                                                         on='transaction_datetime') \
    #                                                                ['return_dollar_amount'].sum().values

    #     transactions_df['mean_account_balance_3d'] = transactions_df.groupby('business_account_number') \
    #                                                                 .rolling('3d', min_periods=1,
    #                                                                          on='transaction_datetime') \
    #                                                                 ['real_ending_balance'].mean().values
    #     transactions_df['mean_account_balance_10d'] = transactions_df.groupby('business_account_number') \
    #                                                                  .rolling('10d', min_periods=1,
    #                                                                           on='transaction_datetime') \
    #                                                                  ['real_ending_balance'].mean().values
    #     transactions_df['mean_account_balance_30d'] = transactions_df.groupby('business_account_number') \
    #                                                                  .rolling('30d', min_periods=1,
    #                                                                           on='transaction_datetime') \
    #                                                                  ['real_ending_balance'].mean().values

    #     transactions_df['std_account_balance_3d'] = transactions_df.groupby('business_account_number') \
    #                                                                .rolling('3d', min_periods=1,
    #                                                                         on='transaction_datetime') \
    #                                                                ['real_ending_balance'].std().values
    #     transactions_df['std_account_balance_10d'] = transactions_df.groupby('business_account_number') \
    #                                                                 .rolling('10d', min_periods=1,
    #                                                                          on='transaction_datetime') \
    #                                                                 ['real_ending_balance'].std().values
    #     transactions_df['std_account_balance_30d'] = transactions_df.groupby('business_account_number') \
    #                                                                 .rolling('30d', min_periods=1,
    #                                                                          on='transaction_datetime') \
    #                                                                 ['real_ending_balance'].std().values

    #     transactions_df['deposit_transaction_amount'] = (transactions_df['is_deposit'] * transactions_df['transaction_amount']).replace(0, np.nan)

    #     transactions_df = pd.merge(transactions_df,
    #                   transactions_df.groupby('business_account_number')['deposit_transaction_amount'].first()\
    #                   .rename('first_deposit_amount').to_frame(),
    #                   how='left', on='business_account_number')

    #     transactions_df = pd.merge(transactions_df,
    #                   transactions_df.groupby('business_account_number')['deposit_transaction_amount'].last()\
    #                   .rename('last_deposit_amount').to_frame(),
    #                   how='left', on='business_account_number')

    #     transactions_df['sum_deposits_3d'] = transactions_df.groupby('business_account_number') \
    #                                                         .rolling('3d', min_periods=1,
    #                                                                  on='transaction_datetime') \
    #                                                         ['deposit_transaction_amount'].sum().values
    #     transactions_df['sum_deposits_10d'] = transactions_df.groupby('business_account_number') \
    #                                                          .rolling('10d', min_periods=1,
    #                                                                   on='transaction_datetime') \
    #                                                          ['deposit_transaction_amount'].sum().values
    #     transactions_df['sum_deposits_30d'] = transactions_df.groupby('business_account_number') \
    #                                                          .rolling('30d', min_periods=1,
    #                                                                   on='transaction_datetime') \
    #                                                          ['deposit_transaction_amount'].sum().values

    #     transactions_df['is_withdrawal'] = transactions_df['transaction_code'].isin(withdrawal_transaction_codes) & \
    #                                        (transactions_df['transaction_amount'] < 0)
    #     transactions_df['withdrawal_transaction_amount'] = (transactions_df['is_withdrawal'] * transactions_df['transaction_amount']).replace(0, np.nan)
    #     transactions_df['sum_withdrawals_3d'] = transactions_df.groupby('business_account_number') \
    #                                                            .rolling('3d', min_periods=1,
    #                                                                     on='transaction_datetime') \
    #                                                            ['withdrawal_transaction_amount'].sum().values
    #     transactions_df['sum_withdrawals_10d'] = transactions_df.groupby('business_account_number') \
    #                                                             .rolling('10d', min_periods=1,
    #                                                                      on='transaction_datetime') \
    #                                                             ['withdrawal_transaction_amount'].sum().values
    #     transactions_df['sum_withdrawals_30d'] = transactions_df.groupby('business_account_number') \
    #                                                             .rolling('30d', min_periods=1,
    #                                                                      on='transaction_datetime') \
    #                                                             ['withdrawal_transaction_amount'].sum().values

    #     transactions_df['mean_deposits_3d'] = transactions_df.groupby('business_account_number') \
    #                                                          .rolling('3d', min_periods=1,
    #                                                                   on='transaction_datetime') \
    #                                                          ['deposit_transaction_amount'].mean().values
    #     transactions_df['mean_deposits_10d'] = transactions_df.groupby('business_account_number') \
    #                                                           .rolling('10d', min_periods=1,
    #                                                                    on='transaction_datetime') \
    #                                                           ['deposit_transaction_amount'].mean().values
    #     transactions_df['mean_deposits_30d'] = transactions_df.groupby('business_account_number') \
    #                                                           .rolling('30d', min_periods=1,
    #                                                                    on='transaction_datetime') \
    #                                                           ['deposit_transaction_amount'].mean().values

    #     transactions_df['mean_withdrawals_3d'] = transactions_df.groupby('business_account_number') \
    #                                                             .rolling('3d', min_periods=1,
    #                                                                       on='transaction_datetime') \
    #                                                             ['withdrawal_transaction_amount'].mean().values
    #     transactions_df['mean_withdrawals_10d'] = transactions_df.groupby('business_account_number') \
    #                                                              .rolling('10d', min_periods=1,
    #                                                                       on='transaction_datetime') \
    #                                                              ['withdrawal_transaction_amount'].mean().values
    #     transactions_df['mean_withdrawals_30d'] = transactions_df.groupby('business_account_number') \
    #                                                              .rolling('30d', min_periods=1,
    #                                                                       on='transaction_datetime') \
    #                                                              ['withdrawal_transaction_amount'].mean().values

    #     transactions_df['max_deposits_3d'] = transactions_df.groupby('business_account_number') \
    #                                                         .rolling('3d', min_periods=1,
    #                                                                   on='transaction_datetime') \
    #                                                         ['deposit_transaction_amount'].max().values
    #     transactions_df['max_deposits_10d'] = transactions_df.groupby('business_account_number') \
    #                                                          .rolling('10d', min_periods=1,
    #                                                                   on='transaction_datetime') \
    #                                                          ['deposit_transaction_amount'].max().values
    #     transactions_df['max_deposits_30d'] = transactions_df.groupby('business_account_number') \
    #                                                          .rolling('30d', min_periods=1,
    #                                                                   on='transaction_datetime') \
    #                                                          ['deposit_transaction_amount'].max().values

    #     transactions_df['max_withdrawals_3d'] = transactions_df.groupby('business_account_number') \
    #                                                            .rolling('3d', min_periods=1,
    #                                                                     on='transaction_datetime') \
    #                                                            ['withdrawal_transaction_amount'].max().values
    #     transactions_df['max_withdrawals_10d'] = transactions_df.groupby('business_account_number') \
    #                                                             .rolling('10d', min_periods=1,
    #                                                                      on='transaction_datetime') \
    #                                                             ['withdrawal_transaction_amount'].max().values
    #     transactions_df['max_withdrawals_30d'] = transactions_df.groupby('business_account_number') \
    #                                                             .rolling('30d', min_periods=1,
    #                                                                      on='transaction_datetime') \
    #                                                             ['withdrawal_transaction_amount'].max().values

    #     transactions_df['is_dd'] = transactions_df['transaction_code'] == 'ACHINDD'
    #     transactions_df['dd_dollar_amount'] = transactions_df['is_dd'] * transactions_df['transaction_amount']

    #     transactions_df['nr_direct_deposits'] = transactions_df.groupby('business_account_number')['is_dd'].cumsum()
    #     transactions_df['dollar_val_dd'] = transactions_df.groupby('business_account_number')['dd_dollar_amount'].cumsum()

    transactions_df = pd.merge(
        transactions_df,
        transactions_df.groupby("business_account_number")["business_account_number"]
        .count()
        .rename("nr_transactions_all_time")
        .reset_index(),
        how="inner",
        on="business_account_number",
    )

    return transactions_df


def process_user_metadata_dw(user_metadata_dw_df, **kwargs):
    """
    Process user metadata from dw.
    """
    print("Processing user metadata from dw.")
    return user_metadata_dw_df
