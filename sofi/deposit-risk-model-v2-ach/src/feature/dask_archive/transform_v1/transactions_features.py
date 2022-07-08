import gc
import pandas as pd
import dask.dataframe as dd
from multiprocessing import Pool, cpu_count

def assign_column(df, col_name, df_to_assign, id_cols=["business_account_number", "transaction_id"]):
    context={col_name: df.head().apply(lambda x: df_to_assign.loc[tuple([x[id_col] for id_col in id_cols])], axis=1)}
    meta = df.head().assign(**context)[:0]
    df = df.map_partitions(lambda ddd: ddd.assign(**context), meta=meta)
    return df

def get_transactions_features(df, id_col=None, dt_col=None, idx_col=None, dup_token="_<DUP>"):
    ### TRANSACTION (not roll-ups) FEATURES

    df["trnx_transaction_as_pct_of_balance"] = df["trnx_transaction_amount"] / (df["trnx_real_ending_balance"] - df["trnx_transaction_amount"])

    return df