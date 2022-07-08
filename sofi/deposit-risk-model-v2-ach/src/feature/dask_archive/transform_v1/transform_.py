# feature transform vision 1 currently used for
# deposit v1, deposit v2, customer risk
import dask.dataframe as dd
from src.feature.dask_archive.transform_v1.external_account_linkage import link_external_accounts
from src.feature.dask_archive.transform_v1.transactions_features import get_transactions_features


def transform(df):

    # set and created sorted id
    # df["datetime_str"] = df.transaction_datetime.astype(int)
    df["transaction_id"] = (
        df["business_account_number"].astype(str) + "-" + df["tseq"].astype(str).apply(lambda x:x.zfill(15))
    )
    # divisions = sorted(df["business_account_number"].unique().compute().astype(str).tolist())
    df = df.set_index("transaction_id", sorted=True)
    # Original id definition.
    # df["group_rank"] = df.groupby("business_account_number")["transaction_datetime"].rank("first").astype(int)
    # df = df.reset_index()
    # df["transaction_id"] = df["transaction_id"] + "-" + df["group_rank"]

    # meta
    id_col = "business_account_number"
    dt_col = "transaction_datetime"
    idx_col = "transaction_id"
    original_cols = df.columns.tolist()
    exist_cols = df.columns.tolist()  # to keep track new features
    content = {"original_cols": original_cols}

    # ==================
    # Banking features
    # ==================
    df = dd.merge(
        df,
        df.groupby(id_col)[dt_col]
        .min()
        .rename("banking_first_transaction_datetime")
        .to_frame(),
        how="left",
        left_on=id_col,
        right_index=True
    )

    df["banking_days_since_first_deposit"] = (
        df[dt_col] - df["banking_first_transaction_datetime"]
    ).dt.days
    df["banking_age_money_account"] = (
        df[dt_col] - df["banking_acct_open_date"]
    ).dt.days
    df["banking_lag_acc_open_first_transaction"] = (
        df["banking_first_transaction_datetime"] - df["banking_acct_open_date"]
    ).dt.days
    exist_cols, content = register_features(
        df.columns, exist_cols, "banking_features", content
    )

    # ==================
    # Credit features
    # ==================

    # ==================
    # Giact features
    # ==================
    df["giact_time_since_first_link"] = (
        df[dt_col] - df["giact_first_link_date"]
    ).dt.days
    df["giact_time_since_last_link"] = (df[dt_col] - df["giact_last_link_date"]).dt.days
    exist_cols, content = register_features(
        df.columns, exist_cols, "giact_features", content
    )

    # ==================
    #  TMX features
    # ==================
    # =========================
    #  Account linkage features
    # =========================

    df, _ = link_external_accounts(df)

    exist_cols, content = register_features(
        df.columns, exist_cols, "external_acct_features", content
    )

    # ==================
    #  Transactions features
    # ==================
    df = get_transactions_features(df, id_col, dt_col, idx_col)
    exist_cols, content = register_features(
        df.columns, exist_cols, "trnx_features", content
    )

    # df = get_deposit_v1_features(df, id_col, dt_col, idx_col)
    # exist_cols, content = register_features(
    #     df.columns, exist_cols, "deposit_v1_features", content
    # )
    #
    # df = get_debugging_features(df, id_col, dt_col, idx_col)
    # exist_cols, content = register_features(
    #     df.columns, exist_cols, "debugging_features", content
    # )
    # ==================
    #  Other features
    # ==================

    return df, content


def register_features(all_cols, exist_cols, name, content):
    """
    register newly added features = all_cols \ exist_cols
    into content[name]

    return updated exist_cols, and content dictionary
    """
    new_cols = all_cols[~all_cols.isin(exist_cols)].tolist()
    exist_cols = all_cols
    content[name] = new_cols
    return exist_cols, content





