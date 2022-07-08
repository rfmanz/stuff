def get_deposit_v1_features(
    df, id_col=None, dt_col=None, idx_col=None, dup_token="_<DUP>"
):
    ################################################
    #      DEPOSIT V1 features for bencharking:
    ################################################

    # refactor rolling with dask
    df["trnx_rolling_mean_acc_bal"] = df.groupby(id_col).rolling("14d", min_periods=1, on=dt_col)["trnx_real_ending_balance"].mean().values
    # df["trnx_nr_past_transactions"] = df.groupby(id_col)[id_col].cumcount()

    return df