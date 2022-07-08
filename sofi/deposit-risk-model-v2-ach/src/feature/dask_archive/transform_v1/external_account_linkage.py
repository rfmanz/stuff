import pandas as pd
import dask.dataframe as dd

def link_external_accounts(transactions_df):
    from tqdm import tqdm
    ### External bank linkages:
    res = []
    curr = None
    counter = {}

    for idx, row in tqdm(
        transactions_df[
            [
                "business_account_number",
                "transaction_datetime",
                "trnx_external_account_number",
                "trnx_external_institution_id",
                "trnx_transaction_amount",
            ]
        ].iterrows()
    ):
        out = [idx]
        row = row.values

        if row[0] != curr:
            curr = row[0]
            counter = {}

        if not row[2]:
            res.append(out + [None for i in range(6)])
            continue

        external_account_number = row[2]

        if external_account_number not in counter:
            counter[external_account_number] = {}

        # nr past transactions with this account
        if "nr_trans_with_acc" in counter[external_account_number]:
            counter[external_account_number]["nr_trans_with_acc"] += 1
        else:
            counter[external_account_number]["nr_trans_with_acc"] = 1
        out.append(counter[external_account_number]["nr_trans_with_acc"] - 1)

        # first transaction dt
        if "first_transaction_dt" not in counter[external_account_number]:
            counter[external_account_number]["first_transaction_dt"] = row[1]
        out.append(counter[external_account_number]["first_transaction_dt"])

        # last transaction_dt
        if "last_transaction_dt" not in counter[external_account_number]:
            counter[external_account_number]["last_transaction_dt"] = None
        out.append(counter[external_account_number]["last_transaction_dt"])
        counter[external_account_number]["last_transaction_dt"] = row[1]

        # sum pos/neg transactions with acct
        if "sum_pos_trans" not in counter[external_account_number]:
            counter[external_account_number]["sum_pos_trans"] = 0
        if "sum_neg_trans" not in counter[external_account_number]:
            counter[external_account_number]["sum_neg_trans"] = 0
        out.append(counter[external_account_number]["sum_pos_trans"])
        out.append(counter[external_account_number]["sum_neg_trans"])

        if row[4] >= 0:
            counter[external_account_number]["sum_pos_trans"] += row[4]
        else:
            counter[external_account_number]["sum_neg_trans"] += row[4]

        if "rolling_mean_pos_trans" not in counter[external_account_number]:
            counter[external_account_number]["rolling_mean_pos_trans"] = row[4]
            out.append(None)
        else:
            out.append(counter[external_account_number]["rolling_mean_pos_trans"])
            counter[external_account_number]["rolling_mean_pos_trans"] = (
                counter[external_account_number]["rolling_mean_pos_trans"] + row[4]
            ) / 2

        res.append(out)

    ea_cols = [
        "transaction_id",
        "ea_nr_trans_with_acc",
        "ea_first_trans_with_dt",
        "ea_last_trans_with_dt",
        "ea_sum_pos_trans",
        "ea_sum_neg_trans",
        "ea_rolling_mean_pos_trans",
    ]


    ea_df = dd.from_pandas(pd.DataFrame(res, columns=ea_cols), npartitions=1)
    # ea_df.columns = ea_cols
    ea_df = ea_df.set_index("transaction_id", sorted=True)
    ea_df["ea_first_trans_with_dt"] = dd.to_datetime(ea_df["ea_first_trans_with_dt"]).dt.tz_localize(None)
    ea_df["ea_last_trans_with_dt"] = dd.to_datetime(ea_df["ea_last_trans_with_dt"]).dt.tz_localize(None)
    ea_df["ea_nr_trans_with_acc"] = ea_df["ea_nr_trans_with_acc"].astype(float)
    ea_df["ea_sum_pos_trans"] = ea_df["ea_sum_pos_trans"].astype(float)
    ea_df["ea_sum_neg_trans"] = ea_df["ea_sum_neg_trans"].astype(float)
    ea_df["ea_rolling_mean_pos_trans"] = ea_df["ea_rolling_mean_pos_trans"].astype(float)

    transactions_df = dd.merge(transactions_df, ea_df, left_index=True, right_index=True, how="inner",
                               suffixes=("", "_<DUP>"))
    # transactions_df_ = dd.concat([transactions_df, ea_df], axis=0)
    # transactions_df = transactions_df.assign(**dict.fromkeys(ea_cols, np.nan))
    # res = dd.from_array(np.array(res))
    # transactions_df[ea_cols] = res

    transactions_df["ea_time_since_first_trans"] = (
            transactions_df["transaction_datetime"] - transactions_df["ea_first_trans_with_dt"]
    ).dt.days
    transactions_df["ea_time_since_last_trans"] = (
            transactions_df["transaction_datetime"] - transactions_df["ea_last_trans_with_dt"]
    ).dt.days

    transactions_df["ea_ratio_all_trans_div_tamt"] = (
            transactions_df["ea_sum_pos_trans"] / transactions_df["trnx_transaction_amount"]
    )
    transactions_df["ea_ratio_rolling_mean_tamt_div_tamt"] = (
            transactions_df["ea_rolling_mean_pos_trans"] / transactions_df["trnx_transaction_amount"]
    )

    # import pdb; pdb.set_trace()
    del res
    return transactions_df, ea_cols
