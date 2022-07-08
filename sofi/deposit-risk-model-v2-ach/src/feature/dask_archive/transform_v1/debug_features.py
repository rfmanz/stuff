import dask.dataframe as dd

def get_debugging_features(df, id_col=None, dt_col=None, idx_col=None, dup_token="_<DUP>"):
    ######################################################
    #    FEATURES FOR DEBUGGING - has data snooping bias
    ######################################################

    df = dd.merge(
        df,
        df.groupby(id_col)[id_col]
            .count()
            .rename("debug_nr_transactions_all_time")
            .reset_index(),
        how="inner",
        on=id_col,
        suffixes=("", dup_token),
    )
    df = dd.merge(
        df,
        df[df["trnx_is_return"]]
            .groupby(id_col)[dt_col]
            .min()
            .rename("debug_first_return_date")
            .reset_index(),
        left="left",
        on=id_col,
    )
    df = dd.merge(
        df,
        df[df["trnx_transaction_code"].isin(["DDCHGOFF", "DDWRTOFF", "DDFRDWO"])]
            .groupby(id_col)[dt_col]
            .min()
            .rename("debug_chg_wrt_off_date")
            .reset_index(),
        how="left",
        on=id_col,
    )
    df = dd.merge(
        df,
        df.groupby(id_col)["trnx_id_return"]
            .sum()
            .rename("debug_nr_returns_all_time")
            .reset_index(),
        how="inner",
        on=id_col,
    )

    return df