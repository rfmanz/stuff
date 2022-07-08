import numpy as np
import pandas as pd
from src.feature.utils import applyParallel
from tqdm import tqdm
from pandas.tseries.offsets import BDay
import gc

def sample_tdf(transactions_df):
    # for debugging purposes
    
    debug = False
    sample = True
    id_col = 'business_account_number'
    id_token = "<BUSINESS_ACCOUNT_NUMBER>"
    if debug:
        debug_ids = pd.read_csv('../../../artifacts/debug_ids.csv')
        debug_ids[id_token] = debug_ids[id_token].astype(transactions_df[id_col].dtype)
        transactions_df = transactions_df[transactions_df[id_col].isin(debug_ids[id_token])]
        # load ids
        # filter by ids
    # debug version: load and filter by debug_ids 
    elif sample:
        transactions_df = transactions_df.sample(n=1000000, replace=False)
        
    return transactions_df


def get_transactions_id(transactions_df):
    transactions_df.sort_values(by=['business_account_number', 
                                    'transaction_datetime'],
                                inplace=True)
    transactions_df['group_rank'] = transactions_df.groupby('business_account_number')['transaction_datetime'].rank('first').astype(int)

    # get id by combining bid, datatime, and rank
    # actually need a way to order this key...
    transactions_df['transaction_id'] = (transactions_df['business_account_number'].astype(str) 
                                         + '-' + transactions_df['transaction_datetime'].apply(lambda x: str(int(x.timestamp())))
                                         + '-' + transactions_df['group_rank'].astype(str))
    return transactions_df


def func(df):
    raise NotImplemented("Use as a place holder for func in applyParallel")

######################################################
#        Helper functions for ACH Labeling
######################################################

def reverse_df(df):
    """ Helper for forward looking rolling function """
    # reverse dataset
    reverse_df = df.iloc[::-1]
    ri = reverse_df.index

    # re-reverse index
    reverse_df.index = ri[0] - ri + ri[-1]

    return reverse_df

## Need to define forward looking roll-ups at top level to parallelize.
def get_nr_returns_5d(df):
    return reverse_df(reverse_df(df).rolling('5d', min_periods=1)['is_return'].sum())

def get_bal_after_5d(df):
    return reverse_df(reverse_df(df).rolling('5d', min_periods=1)['trnx_real_ending_balance'].apply(lambda a: a[0], raw=True))

def match_ach_returns(df, timedelta=BDay(3)):
    """
    Check if ACH transactions are returned with some timeframe.
    """
    #For performance purposes...
    df = df[df['trnx_transaction_code'].isin(['ACHDD', 'DWACHRET'])]
    
    # df = df.reset_index(drop=True)

    df['tamt_abs'] = df['trnx_transaction_amount'].abs()
    df['is_returned'] = False
    
    for act_n, transaction in tqdm(df[df['is_return'] == 1].iterrows()):
        
        tdt_hi = transaction['transaction_datetime']
        tdt_lo = tdt_hi - timedelta

        dcandidate = df[df['business_account_number'] == transaction['business_account_number']]

        dret = dcandidate[(dcandidate['transaction_datetime'].between(tdt_lo, tdt_hi)) & \
                          (dcandidate['tamt_abs'] == transaction['tamt_abs'])]
        
        df.loc[dret.index, 'is_returned'] = True

#     df = df.reset_index(drop=True)
    return df

    
def get_labels(df):
    """
    Get add labels to processed data.
    """    
    global func
    # sort data for roll-ups
    df = df.sort_values(by=['business_account_number', 'transaction_datetime'])

    df['days_to_acc_close'] = (pd.to_datetime(df['banking_date_acct_closed']) - df['transaction_datetime']).dt.days
    df['account_closed_by_risk_in_next_90d'] = df['banking_closed_reason'].isin(['Closed by SoFi - Risk Request', 
                                                                         'Closed by SoFi - Charge-Off / Write-Off']) &\
                                               (df['days_to_acc_close'] <= 90)
    
    # does account chg/wrt off in next 90 days?
    df['is_chg_wrt_off_in_90d'] = (df['chg_wrt_off_date'] - df['transaction_datetime']).dt.days <= 90

    # Set index to transaction datetime.
    df = df.set_index('transaction_datetime')

    # get num returns by borrower in the next 90 days
    df['nr_returns_in_next_5d'] = applyParallel(df[['business_account_number', 'is_return']].groupby('business_account_number'), get_nr_returns_5d).values
    
    # get this borrower's account balance after 90 days
    df['bal_after_5d'] = applyParallel(df[['business_account_number', 'trnx_real_ending_balance']].groupby('business_account_number'), get_bal_after_5d).values

    df = df.reset_index()
    
    if 'level_0' in df.columns:
        df = df.drop('level_0', axis=1)
    
    return df  

def combine_df_by_idx(df1, df2):
    for col in df2.columns:
        idx = df2.index
        if col in df1.columns:
            print(f"column {col} equals? {df1.loc[idx, col].equals(df2[col])}")
        else:
            print(f"adding column {col}")
            df1[col] = np.nan
            df1.loc[idx, col] = df2[col]
    return df1

# drop non ACH types
def drop_non_ach(df):
    df = df[df['trnx_transaction_code'].isin(['ACHDD']) & (df['transaction_amount'] > 0)]
    return df


######################################################
#        Helper functions for MCD Labeling
######################################################

NDAYS = 90

def get_nr_returns_nd(df):
    global NDAYS
    return reverse_df(reverse_df(df).rolling(f'{NDAYS}d', min_periods=1)['is_return'].sum())

def get_bal_after_nd(df):
    global NDAYS
    return reverse_df(reverse_df(df).rolling(f'{NDAYS}d', min_periods=1)['trnx_real_ending_balance'].apply(lambda a: a[0], raw=True))


def get_labels_nd(df, n_days=90):
    """
    Get add labels to processed data.
    """
    # sort data for roll-ups
    global NDAYS
    global func
    NDAYS = n_days
    print(f'Ndays = {NDAYS}')
    
    # sort data for roll-ups
    
#     df['days_to_acc_close'] = (pd.to_datetime(df['dtc']) - df['transaction_datetime']).dt.days
    df[f'account_closed_by_risk_in_next_{n_days}d'] = df['banking_closed_reason'].isin(['Closed by SoFi - Risk Request', 
                                                                         'Closed by SoFi - Charge-Off / Write-Off']) &\
                                               (df['days_to_acc_close'] <= n_days)
    
#     df['last_unrestricted_date_in_next_90d'] = (df['last_unrestricted_date'] - df['transaction_datetime']).dt.days.between(0, 90)
    
    # get most recent account balance
    df = pd.merge(df, 
                  df.groupby('business_account_number')['trnx_real_ending_balance'].last().rename('latest_acc_bal').reset_index(),
                  how='left', on='business_account_number')

    # does account chg/wrt off in next n days?
    df[f'is_chg_wrt_off_in_{n_days}d'] = (df['chg_wrt_off_date'] - df['transaction_datetime']).dt.days <= n_days

    # Set index to transaction datetime.
    df = df.set_index('transaction_datetime')

    df[f'bal_after_{n_days}d'] = applyParallel(df[['business_account_number', 
                                           'trnx_real_ending_balance']].groupby('business_account_number'), 
                                                     get_bal_after_nd).values
    df[f'nr_returns_in_next_{n_days}d'] = applyParallel(df[['business_account_number', 
                                                    'is_return']].groupby('business_account_number'), 
                                                 get_nr_returns_nd).values
    df = df.reset_index()
                
    # drop non check types
    
    
    def get_target(df):
        """
        """
        df[f'target_{n_days}d'] = df[f'is_chg_wrt_off_in_{n_days}d'] | \
                       df[f'account_closed_by_risk_in_next_{n_days}d'] | \
                       (df[f'nr_returns_in_next_{n_days}d'] > 0) | \
                       (df[f'bal_after_{n_days}d'] < 0)
        
        df[f'indeterminate_{n_days}d'] = (df[f'target_{n_days}d'] & (df[f'bal_after_{n_days}d'] > 0)) | \
                              (~df[f'target_{n_days}d'] & (df[f'bal_after_{n_days}d'] <= 0))
        
        return df

    df = get_target(df)

    return df #.reset_index()

def sort_dfs(df):
    df.sort_values(by=['business_account_number', 'transaction_datetime', 'trnx_transaction_amount'], inplace=True)

def drop_non_check(df):
    df = df[df['trnx_transaction_code'].isin(['DDCK']) & (df['trnx_transaction_amount'] > 0)]
    return df


def transform(tdf):
    global func
    
    tdf = get_transactions_id(tdf)

    # set up id and date columns
    id_col = "business_account_number"
    date_col = "transaction_datetime"

    # sort data by id and then datetime
    tdf = tdf.sort_values(by=[id_col, date_col])
        
    ######################################################
    #              Account level features
    ######################################################
    
    first_tdt = tdf.groupby(id_col)[date_col].min().rename("first_transaction_datetime").to_frame()
    tdf = pd.merge(tdf, first_tdt, how="left", on=id_col)

    tdf["days_since_first_deposit"] = (tdf["transaction_datetime"] - tdf["first_transaction_datetime"]).dt.days
    
    tdf["age_money_account"] = (tdf["transaction_datetime"] - tdf["banking_acct_open_date"]).dt.days
    tdf['lag_acc_open_first_transaction'] = (tdf['first_transaction_datetime'] - \
                                             tdf['banking_acct_open_date']).dt.days

    # first_deposit_amount -> banking_first_deposit_amount
    # External account linkage
    
    ######################################################
    #              External bank linkages
    ######################################################
    
    res = []
    curr = None
    counter = {}

    for row in tqdm(tdf[['business_account_number', 'transaction_datetime', 
                         'trnx_external_account_number', 'trnx_external_institution_id', 
                         'trnx_transaction_amount']].values):
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
        if 'nr_trans_with_acc' in counter[external_account_number]:
            counter[external_account_number]['nr_trans_with_acc'] += 1
        else:
            counter[external_account_number]['nr_trans_with_acc'] = 1
        out.append(counter[external_account_number]['nr_trans_with_acc'] - 1)

        # first transaction dt
        if 'first_transaction_dt' not in counter[external_account_number]:
            counter[external_account_number]['first_transaction_dt'] = row[1]
        out.append(counter[external_account_number]['first_transaction_dt'])

        # last transaction_dt
        if 'last_transaction_dt' not in counter[external_account_number]:
            counter[external_account_number]['last_transaction_dt'] = None
        out.append(counter[external_account_number]['last_transaction_dt'])
        counter[external_account_number]['last_transaction_dt'] = row[1]

        # sum pos/neg transactions with acct
        if 'sum_pos_trans' not in counter[external_account_number]:
            counter[external_account_number]['sum_pos_trans'] = 0
        if 'sum_neg_trans' not in counter[external_account_number]:
            counter[external_account_number]['sum_neg_trans'] = 0
        out.append(counter[external_account_number]['sum_pos_trans'])
        out.append(counter[external_account_number]['sum_neg_trans'])

        if row[4] >= 0:
            counter[external_account_number]['sum_pos_trans'] += row[4]
        else:
            counter[external_account_number]['sum_neg_trans'] += row[4]

        if 'rolling_mean_pos_trans' not in counter[external_account_number]:
            counter[external_account_number]['rolling_mean_pos_trans'] = row[4]
            out.append(None)
        else:
            out.append(counter[external_account_number]['rolling_mean_pos_trans'])
            counter[external_account_number]['rolling_mean_pos_trans'] = (counter[external_account_number]['rolling_mean_pos_trans'] + row[4]) / 2

        res.append(out)

    ea_cols = ['nr_trans_with_acc', 'first_trans_with_ea_dt', 'last_trans_with_ea_dt', 'sum_pos_trans_ea', 'sum_neg_trans_ea', 'rolling_mean_pos_trans_ea']
    tdf = tdf.assign(**dict.fromkeys(ea_cols, np.nan))
    tdf[ea_cols] = res

    del res
    
    
    # ea -> external account
    tdf['time_since_first_trans_ea'] = (tdf['transaction_datetime'] - tdf['first_trans_with_ea_dt']).dt.days
    tdf['time_since_last_trans_ea'] = (tdf['transaction_datetime'] - tdf['last_trans_with_ea_dt']).dt.days

    tdf['ratio_all_ea_trans_div_tamt'] = tdf['sum_pos_trans_ea'] / tdf['trnx_transaction_amount']
    tdf['ratio_rolling_mean_ea_tamt_div_tamt'] = tdf['rolling_mean_pos_trans_ea'] / tdf['trnx_transaction_amount']

    
    ######################################################
    #              Transaction features
    ######################################################
    
    
    ### TRANSACTION (not roll-ups) FEATURES
    tdf['transaction_as_pct_of_balance'] = tdf['trnx_transaction_amount'] / \
                                           (tdf['trnx_real_ending_balance'] - \
                                            tdf['trnx_transaction_amount'])

    tdf['last_transaction_datetime'] = tdf.groupby(id_col)[date_col].shift(1)

    # # this feature doesn't make sense... Just to keep here for record
    tdf['last_transaction_code'] = tdf.groupby(id_col)['trnx_transaction_code'].shift(1)  

    tdf['time_since_last_transaction'] = (tdf['transaction_datetime'] - 
                                          tdf['last_transaction_datetime']).dt.seconds # this relies on transactions we don't like not being included!
    # # transaction features
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

    tdf["is_return"] = tdf["trnx_transaction_code"].isin(
        ["DWCKCB", "DWACHRET", "DDACHRET"]
    ) | (
        (tdf["trnx_transaction_code"].isin(deposit_transaction_codes))
        & (tdf["trnx_transaction_amount"] < 0)
    )
    tdf["is_trans"] = tdf["trnx_transaction_code"].isin(
        deposit_transaction_codes + withdrawal_transaction_codes
    )
    tdf["is_deposit"] = tdf["trnx_transaction_code"].isin(deposit_transaction_codes) & (
        tdf["trnx_transaction_amount"] > 0
    )
    
    tdf['nr_past_returns'] = tdf.groupby(id_col)['is_return'].cumsum()

    def func(df_): return df_.rolling('3d',min_periods=1,on=date_col)['is_deposit'].sum()
    tdf['nr_deposits_3d']=applyParallel(tdf.groupby(id_col), func).values
    
    def func(df_): return df_.rolling('30d',min_periods=1,on=date_col)['is_return'].sum()
    tdf['nr_returns_30d']=applyParallel(tdf.groupby(id_col), func).values

    tdf['nr_past_deposits'] = tdf.groupby(id_col)['is_deposit'].cumsum()

    def func(df_): return df_.rolling('30d',min_periods=1,on=date_col)['is_deposit'].sum()
    tdf['nr_deposits_30d']=applyParallel(tdf.groupby(id_col), func).values

    def func(df_): return df_.rolling('3d',min_periods=1,on=date_col)['is_trans'].sum()
    tdf['nr_transactions_3d']=applyParallel(tdf.groupby(id_col), func).values

    def func(df_): return df_.rolling('30d',min_periods=1,on=date_col)['is_trans'].sum()
    tdf['nr_transactions_30d']=applyParallel(tdf.groupby(id_col), func).values

    tdf['pct_returned_deposits'] = tdf['nr_past_returns'] / tdf['nr_past_deposits']
    tdf['pct_returned_deposits_30d'] = tdf['nr_returns_30d'] / tdf['nr_deposits_30d']
    
    # features cannot be parallelized that may take a while
    # as it turns out cumcount/cumsum/cum* functions are incredibly efficient
    tdf['nr_past_transactions'] = tdf.groupby(id_col)[id_col].cumcount()

    tdf['nr_transactions_30d_div_nr_past_transactions'] = tdf['nr_transactions_30d'] / tdf['nr_past_transactions']

    # features based on account balances
    tdf['tamt_adjusted'] = tdf['trnx_transaction_amount'] * np.where(tdf['trnx_transaction_code'] == 'ACHDD', -1, 1)

    def func(df_): return df_.rolling('3d',min_periods=1,on=date_col)['trnx_real_ending_balance'].mean()
    tdf['mean_account_balance_3d']=applyParallel(tdf.groupby(id_col), func).values

    def func(df_): return df_.rolling('30d',min_periods=1,on=date_col)['trnx_real_ending_balance'].mean()
    tdf['mean_account_balance_30d']=applyParallel(tdf.groupby(id_col), func).values
    
    tdf['deposit_transaction_amount'] = (tdf['is_deposit'] * tdf['trnx_transaction_amount']).replace(np.nan, 0)

    def func(df_): return df_.rolling('3d',min_periods=1,on=date_col)['deposit_transaction_amount'].sum()
    tdf['sum_deposits_3d']=applyParallel(tdf.groupby(id_col), func).values

    def func(df_): return df_.rolling('10d',min_periods=1,on=date_col)['deposit_transaction_amount'].sum()
    tdf['sum_deposits_10d']=applyParallel(tdf.groupby(id_col), func).values

    def func(df_): return df_.rolling('30d',min_periods=1,on=date_col)['deposit_transaction_amount'].sum()
    tdf['sum_deposits_30d']=applyParallel(tdf.groupby(id_col), func).values


    tdf['is_withdrawal'] = tdf['trnx_transaction_code'].isin(withdrawal_transaction_codes) & \
                           (tdf['trnx_transaction_amount'] < 0)
    tdf['withdrawal_transaction_amount'] = (tdf['is_withdrawal'] * tdf['trnx_transaction_amount']).replace(np.nan, 0)

    def func(df_): return df_.rolling('3d',min_periods=1,on=date_col)['withdrawal_transaction_amount'].sum()
    tdf['sum_withdrawals_3d']=applyParallel(tdf.groupby(id_col), func).values

    def func(df_): return df_.rolling('10d',min_periods=1,on=date_col)['withdrawal_transaction_amount'].sum()
    tdf['sum_withdrawals_10d']=applyParallel(tdf.groupby(id_col), func).values

    def func(df_): return df_.rolling('30d',min_periods=1,on=date_col)['withdrawal_transaction_amount'].sum()
    tdf['sum_withdrawals_30d']=applyParallel(tdf.groupby(id_col), func).values
    
    
    def func(df_): return df_.rolling('10d',min_periods=1,on=date_col)['deposit_transaction_amount'].mean()
    tdf['mean_deposits_10d']=applyParallel(tdf.groupby(id_col), func).values

    def func(df_): return df_.expanding().mean()
    tdf['mean_deposits']=applyParallel(tdf.groupby(id_col)['deposit_transaction_amount'], func).values

    def func(df_): return df_.rolling('10d',min_periods=1,on=date_col)['withdrawal_transaction_amount'].mean()
    tdf['mean_withdrawals_10d']=applyParallel(tdf.groupby(id_col), func).values

    def func(df_): return df_.expanding().mean()
    tdf['mean_withdrawals']=applyParallel(tdf.groupby(id_col)['withdrawal_transaction_amount'], func).values
    
    tdf['mean_deposits_10d_div_mean_deposits'] = tdf['mean_deposits_10d'] / tdf['mean_deposits']
    tdf['mean_withdrawals_10d_div_mean_withdrawals'] = tdf['mean_withdrawals_10d'] / tdf['mean_withdrawals']

    def func(df_): return df_.expanding().max()
    tdf['max_deposits']=applyParallel(tdf.groupby(id_col)['deposit_transaction_amount'], func).values

    def func(df_): return df_.rolling('3d',min_periods=1,on=date_col)['deposit_transaction_amount'].max()
    tdf['max_deposits_3d']=applyParallel(tdf.groupby(id_col), func).values
    
    def func(df_): return df_.rolling('10d',min_periods=1,on=date_col)['deposit_transaction_amount'].max()
    tdf['max_deposits_10d']=applyParallel(tdf.groupby(id_col), func).values

    def func(df_): return df_.rolling('30d',min_periods=1,on=date_col)['deposit_transaction_amount'].max()
    tdf['max_deposits_30d']=applyParallel(tdf.groupby(id_col), func).values

    
    def func(df_): return df_.expanding().max()
    tdf['max_withdrawals']=applyParallel(tdf.groupby(id_col)['withdrawal_transaction_amount'], func).values

    def func(df_): return df_.rolling('3d',min_periods=1,on=date_col)['withdrawal_transaction_amount'].min()
    tdf['max_withdrawals_3d']=applyParallel(tdf.groupby(id_col), func).values

    def func(df_): return df_.rolling('10d',min_periods=1,on=date_col)['withdrawal_transaction_amount'].min()
    tdf['max_withdrawals_10d']=applyParallel(tdf.groupby(id_col), func).values

    
    def func(df_): return df_.rolling('30d',min_periods=1,on=date_col)['withdrawal_transaction_amount'].min()
    tdf['max_withdrawals_30d']=applyParallel(tdf.groupby(id_col), func).values

    tdf['max_deposits_10d_div_mean_deposits'] = tdf['max_deposits_10d'] / tdf['mean_deposits']
    tdf['max_deposits_10d_div_mean_account_balance_30d'] = tdf['max_deposits_10d'] / tdf['mean_account_balance_30d']
    tdf['max_withdrawals_10d_div_mean_withdrawals'] = tdf['max_withdrawals_10d'] / tdf['mean_withdrawals']

    tdf['nr_trans_ratio'] = tdf['nr_transactions_3d'] / tdf['nr_transactions_30d']
    tdf['bal_ratio'] = tdf['mean_account_balance_3d'] / tdf['mean_account_balance_30d']
    tdf['deposits_ratio'] = tdf['sum_deposits_3d'] / tdf['sum_deposits_30d']

    tdf['is_dd'] = tdf['trnx_transaction_code'] == 'ACHINDD'
    tdf['dd_dollar_amount'] = tdf['is_dd'] * tdf['trnx_transaction_amount']

    tdf['nr_direct_deposits'] = tdf.groupby(id_col)['is_dd'].cumsum()
    tdf['dollar_val_dd'] = tdf.groupby(id_col)['dd_dollar_amount'].cumsum()

    tdf['return_dollar_amount'] = tdf['is_return'] * tdf['tamt_adjusted']
    tdf['dollar_val_returns'] = tdf.groupby(id_col)['return_dollar_amount'].cumsum()

    def func(df_): return df_.rolling('3d',min_periods=1,on=date_col)['return_dollar_amount'].sum()
    tdf['dollar_val_returns_3d']=applyParallel(tdf.groupby(id_col), func).values

    tdf['plaid_days_since_first_link'] = (tdf[date_col] - tdf['plaid_first_link_date']).dt.days
    
    
    ################################################
    #      DEPOSIT V1 features for bencharking: 
    ################################################
    tdf['account_ending_balance'] = tdf['trnx_real_ending_balance']
    tdf['days_since_first_transaction'] = tdf['days_since_first_deposit']

    def func(df_): return df_.rolling('14d',min_periods=1,on=date_col)['trnx_real_ending_balance'].mean()
    tdf['rolling_mean_acc_bal']=applyParallel(tdf.groupby(id_col), func).values

    # nr_transactions_per_day
    tdf['nr_past_transactions'] = tdf.groupby(id_col)                                            [id_col].cumcount()
    tdf['nr_transactions_per_day'] = tdf['nr_past_transactions'] /                                              tdf['days_since_first_deposit']

    # transaction_as_pct_of_balance
    tdf['transaction_as_pct_of_balance'] = tdf['trnx_transaction_amount'] /                                                    (tdf['trnx_real_ending_balance'] -                                                     tdf['trnx_transaction_amount'])

    # rolling_trns_as_pct_of_bal
    tdf['transaction_as_pct_of_balance_abs'] = tdf['transaction_as_pct_of_balance'].abs()

    def func(df_): return df_.rolling('7d',min_periods=1,on=date_col)['transaction_as_pct_of_balance_abs'].mean()
    tdf['rolling_trns_as_pct_of_bal']=applyParallel(tdf.groupby(id_col), func).values

    # transaction_as_pct_of_bal_min
    def func(df_): return df_.rolling('7d',min_periods=1,on=date_col)['transaction_as_pct_of_balance'].min()
    tdf['transaction_as_pct_of_bal_min']=applyParallel(tdf.groupby(id_col), func).values

    # rolling_mean_acc_bal <<< THIS IS WRONG? something going wrong!!
    def func(df_): return df_.rolling('14d',min_periods=1,on=date_col)['trnx_real_ending_balance'].mean()
    tdf['rolling_mean_acc_bal']=applyParallel(tdf.groupby(id_col), func).values
    
    
    ####################################################################
    #    FEATURES FOR LABELING and DEBUGGING - has data snooping bias
    ####################################################################
    tdf = pd.merge(tdf, 
                   tdf.groupby(id_col)[id_col].count().rename('nr_transactions_all_time').reset_index(),
                   how='inner',
                   on=id_col)

    tdf = pd.merge(tdf, 
                   tdf[tdf['is_return']].groupby(id_col)[date_col].min().rename('first_return_date').reset_index(),
                   how='left',
                   on=id_col)

    tdf = pd.merge(tdf, 
                   tdf[tdf['trnx_transaction_code'].isin(['DDCHGOFF', 'DDWRTOFF', 'DDFRDWO'])].groupby(id_col)[date_col].min().rename('chg_wrt_off_date').reset_index(),
                   how='left',
                   on=id_col)

    tdf = pd.merge(tdf, 
                   tdf.groupby(id_col)['is_return'].sum().rename('nr_returns_all_time').reset_index(),
                   how='inner',
                   on=id_col)

    
    ####################################################################
    #    More features from later iterations
    ####################################################################
    
    # tdf['nr_days_to_chg_wrt_off'] = (tdf['chg_wrt_off_date'] - tdf[date_col]).dt.days


    # number of different types of returns
    tdf['is_return_ach'] = tdf['trnx_transaction_code'].isin(['DWACHRET', 'DDACHRET'])
    tdf['is_return_mcd'] = tdf['trnx_transaction_code'].isin(['DWCKCB'])
    tdf['is_return_other'] = ((tdf['trnx_transaction_code'].isin(deposit_transaction_codes)) &                                        (tdf['trnx_transaction_amount'] < 0))

    # nr_past_return_types
    tdf['nr_past_returns_ach'] = tdf.groupby(id_col)['is_return_ach'].cumsum().values
    tdf['nr_past_returns_mcd'] = tdf.groupby(id_col)['is_return_mcd'].cumsum().values
    tdf['nr_past_returns_other'] = tdf.groupby(id_col)['is_return_other'].cumsum().values

    def func(df_): return df_.rolling('3d',min_periods=1,on=date_col)['is_return'].sum()
    tdf['nr_returns_3d']=applyParallel(tdf.groupby(id_col), func).values

    def func(df_): return df_.rolling('3d',min_periods=1,on=date_col)['is_return_ach'].sum()
    tdf['nr_returns_ach_3d']=applyParallel(tdf.groupby(id_col), func).values

    def func(df_): return df_.rolling('3d',min_periods=1,on=date_col)['is_return_mcd'].sum()
    tdf['nr_returns_mcd_3d']=applyParallel(tdf.groupby(id_col), func).values


    # bouncing streaks
    deposit_transaction_codes = ['POSDD', 'ACHDD', 'ACHDDIN', 'ACHINDD', 'DDCK', 'DDMBR', 'DD']

    condition = (tdf['trnx_transaction_code'].isin(deposit_transaction_codes) |
                 tdf['is_return'])
    df_temp = tdf[condition][['transaction_id', 
                                          id_col, 
                                          date_col,
                                          'is_return']]
    df_temp.sort_values([id_col, date_col], inplace=True)
    # out of past 5 deposit related transactions, how many were returns
    def func(df_): return df_.rolling(5,min_periods=1)['is_return'].mean()
    df_temp['rolling_deposit_returns']=applyParallel(df_temp.groupby(id_col), func).values

    tdf = pd.merge(tdf, df_temp[['transaction_id', 'rolling_deposit_returns']], 
                               how='left', on='transaction_id')
    tdf['rolling_deposit_returns'].fillna(0, inplace=True)

    condition = tdf['deposit_transaction_amount'] > 0
    meta_cols = ['transaction_id', 
                 id_col, 
                 date_col,
                 'deposit_transaction_amount']
    df_temp = tdf[condition][meta_cols]
    
    # captures small, small, large pattern
    def func(df_): return df_.rolling(3,min_periods=1)['deposit_transaction_amount'].median()
    df_temp['median_deposits_last_3']=applyParallel(df_temp.groupby(id_col), func).values

    def func(df_): return df_.rolling(5,min_periods=1)['deposit_transaction_amount'].median()
    df_temp['median_deposits_last_5']=applyParallel(df_temp.groupby(id_col), func).values

    def func(df_): return df_.rolling(10,min_periods=1)['deposit_transaction_amount'].median()
    df_temp['median_deposits_last_10']=applyParallel(df_temp.groupby(id_col), func).values

    def func(df_): return df_.rolling(3,min_periods=1)['deposit_transaction_amount'].mean()
    df_temp['mean_deposits_last_3']=applyParallel(df_temp.groupby(id_col), func).values

    def func(df_): return df_.rolling(5,min_periods=1)['deposit_transaction_amount'].mean()
    df_temp['mean_deposits_last_5']=applyParallel(df_temp.groupby(id_col), func).values

    def func(df_): return df_.rolling(10,min_periods=1)['deposit_transaction_amount'].mean()
    df_temp['mean_deposits_last_10']=applyParallel(df_temp.groupby(id_col), func).values

    def func(df_): return df_.rolling(3,min_periods=1)['deposit_transaction_amount'].max()
    df_temp['max_deposits_last_3']=applyParallel(df_temp.groupby(id_col), func).values

    def func(df_): return df_.rolling(5,min_periods=1)['deposit_transaction_amount'].max()
    df_temp['max_deposits_last_5']=applyParallel(df_temp.groupby(id_col), func).values

    def func(df_): return df_.rolling(10,min_periods=1)['deposit_transaction_amount'].max()
    df_temp['max_deposits_last_10']=applyParallel(df_temp.groupby(id_col), func).values


    df_temp['deposits_trend_short1'] = df_temp['max_deposits_last_3'] - df_temp['median_deposits_last_5']
    df_temp['deposits_trend_ratio_short1'] = df_temp['max_deposits_last_3'] / df_temp['median_deposits_last_5']
    df_temp['deposits_trend_mid1'] = df_temp['max_deposits_last_3'] - df_temp['median_deposits_last_10']
    df_temp['deposits_trend_ratio_mid1'] = df_temp['max_deposits_last_3'] / df_temp['median_deposits_last_10']
    df_temp['deposits_trend_mid2'] = df_temp['mean_deposits_last_3'] - df_temp['median_deposits_last_10']
    df_temp['deposits_trend_ratio_mid2'] = df_temp['mean_deposits_last_3'] / df_temp['median_deposits_last_10']

    new_cols = [f for f in df_temp.columns if f not in meta_cols]
    tdf = pd.merge(tdf, df_temp[new_cols+['transaction_id']], 
                               how='left', on='transaction_id')
    tdf[new_cols].fillna(0, inplace=True)


    # nr_deposits, nr_transactions over a period; and their ratios

    def func(df_): return df_.rolling('24h',min_periods=1,on=date_col)['is_trans'].sum()
    tdf['nr_transactions_1d']=applyParallel(tdf.groupby(id_col), func).values

    def func(df_): return df_.rolling('24h',min_periods=1,on=date_col)['is_deposit'].sum()
    tdf['nr_deposits_1d']=applyParallel(tdf.groupby(id_col), func).values

    tdf['nr_deposits_ratio_1d'] = tdf['nr_deposits_1d'] / tdf['nr_transactions_1d']

    def func(df_): return df_.rolling('3h',min_periods=1,on=date_col)['is_trans'].sum()
    tdf['nr_transactions_3h']=applyParallel(tdf.groupby(id_col), func).values

    def func(df_): return df_.rolling('3h',min_periods=1,on=date_col)['is_deposit'].sum()
    tdf['nr_deposits_3h']=applyParallel(tdf.groupby(id_col), func).values

    tdf['nr_deposits_ratio_3h'] = tdf['nr_deposits_3h'] / tdf['nr_transactions_3h']
    
    ####################################################################
    #    ACH labels
    ####################################################################
    
    cols = """
    business_account_number
    transaction_datetime
    banking_date_acct_closed
    banking_closed_reason
    chg_wrt_off_date
    is_return
    trnx_real_ending_balance
    trnx_transaction_code
    trnx_transaction_amount
    """.split()

    tdf_ = get_labels(tdf[cols])
    tdf__ = match_ach_returns(tdf_)
    
    
    # join labeled dfs with 
    tdf = combine_df_by_idx(tdf, tdf_)
    tdf = combine_df_by_idx(tdf, tdf__)
    
    del tdf_
    del tdf__
    gc.collect()
    
    
    ####################################################################
    #    MCD labels
    ####################################################################
    
    sort_dfs(tdf)

    for n in tqdm([10, 30, 60]):
        tdf = get_labels_nd(tdf, n_days=n)
    
    return tdf, {}
