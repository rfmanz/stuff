"""
Select features for each model.
"""
import json
import os
import time

import lightgbm as lgb
import pandas as pd
import numpy as np

from data import load_dataframe
from rdsutils.boruta import Boruta


src_base = os.path.dirname(os.path.realpath(__file__))
config_file_path = os.path.abspath(os.path.join(src_base, "../config.json"))
with open(config_file_path, "r") as f:
    CONFIG_FILE = json.load(f)


# there HAS to be a better way to do this...
# technical debt I guess...
quovo_plaid_features = [
    "pq_max_avail_bal",
    "pq_min_avail_bal",
    "pq_total_pos_bal",
    "pq_total_neg_bal",
    "pq_largest_checking_or_savings_balance",
    "pq_available_bal",
]

features = [
    "transaction_amount",
    "real_ending_balance",
    "fico_score",
    "vantage_score",
    "all7120",
    "all8220",
    "bcc2800",
    "bcc7120",
    "bcx3423",
    "iln5520",
    "iqt9413",
    "iqt9415",
    "mtf5820",
    "stu5031",
    "credit_card_loan_amount",
    "delinquencies_90_days",
    "education_loan_amount",
    "mortgage_loan_amount",
    "secured_loan_amount",
    "total_outstanding_balance",
    "total_tradelines_open",
    "unsecured_loan_amount",
    "giact_nr_pass",
    "giact_nr_decline",
    "giact_nr_other",
    "fraud_score_1",
    "fraud_score_2",
    "address_risk_score",
    "email_risk_score",
    "phone_risk_score",
    "name_address_correlation",
    "name_email_correlation",
    "name_phone_correlation",
    "nr_social_profiles_found",
    "days_since_first_deposit",
    "age_money_account",
    "lag_acc_open_first_transaction",
    "first_deposit_amount",
    "giact_time_since_first_link",
    "giact_time_since_last_link",
    "nr_trans_with_acc",
    "sum_pos_trans_ea",
    "sum_neg_trans_ea",
    "rolling_mean_pos_trans_ea",
    "time_since_first_trans_ea",
    "time_since_last_trans_ea",
    "ratio_all_ea_trans_div_tamt",
    "ratio_rolling_mean_ea_tamt_div_tamt",
    "transaction_as_pct_of_balance",
    "time_since_last_transaction",
    "nr_past_returns",
    "nr_returns_30d",
    "nr_past_deposits",
    "nr_deposits_3d",
    "nr_deposits_30d",
    "nr_past_transactions",
    "nr_transactions_3d",
    "nr_transactions_30d",
    "pct_returned_deposits",
    "pct_returned_deposits_30d",
    "nr_transactions_30d_div_nr_past_transactions",
    "tamt_adjusted",
    "mean_account_balance_3d",
    "mean_account_balance_30d",
    "std_account_balance_3d",
    "std_account_balance_30d",
    "sum_deposits_3d",
    "sum_deposits_10d",
    "sum_deposits_30d",
    "sum_withdrawals_3d",
    "sum_withdrawals_10d",
    "sum_withdrawals_30d",
    "mean_deposits_10d",
    "mean_deposits",
    "mean_deposits_10d_div_mean_deposits",
    "mean_withdrawals_10d",
    "mean_withdrawals",
    "mean_withdrawals_10d_div_mean_withdrawals",
    "max_deposits",
    "max_deposits_3d",
    "max_deposits_10d",
    "max_deposits_10d_div_mean_deposits",
    "max_deposits_10d_div_mean_account_balance_30d",
    "max_withdrawals",
    "max_withdrawals_3d",
    "max_withdrawals_10d",
    "max_withdrawals_10d_div_mean_withdrawals",
    "nr_trans_ratio",
    "bal_ratio",
    "deposits_ratio",
    "nr_direct_deposits",
    "dd_dollar_amount",
    "dollar_val_dd",
]

features = features + quovo_plaid_features

boruta_args = {
    "drop_at": -15,
    "max_iter": 50,
    "random_state": 10,
    "thresh": 0.3,
    "verbose": 1,
}

lgb_default_params = {
    "boosting_type": "gbdt",
    "metric": "auc",
    "max_depth": 4,
    "n_estimators": 400,
    "colsample_bytree": 0.75,
    "learning_rate": 0.1,
    "lambda_l1": 5,
    "lambda_l2": 25,
    "min_data_in_leaf": 100,
    "random_state": 222,
}


def select_features(X, y, feature_selector):
    """
    X - numpy array
    y - numpy array
    """
    feature_selector.fit(X, y)

    dimp = pd.DataFrame(
        {
            "feature": features,
            "score": feature_selector.scores,
            "mean_imp": np.mean(feature_selector.imps, axis=0),
        }
    ).sort_values(by=["score", "mean_imp"], ascending=False)

    return dimp


def main():
    """ """
    target_col_name = CONFIG_FILE["target_column"]
    indeterminate_col_name = CONFIG_FILE["indeterminate_column"]

    df = load_dataframe("labeled", "labeled")
    df = df[~df[indeterminate_col_name]]

    df["pq_max_avail_bal"] = df[["plaid_max_avail_bal", "quovo_max_avail_bal"]].max(
        axis=1
    )
    df["pq_min_avail_bal"] = df[["plaid_min_avail_bal", "quovo_min_avail_bal"]].min(
        axis=1
    )
    df["pq_total_pos_bal"] = df[["plaid_total_pos_bal", "quovo_total_pos_bal"]].max(
        axis=1
    )
    df["pq_total_neg_bal"] = df[["plaid_total_neg_bal", "quovo_total_neg_bal"]].min(
        axis=1
    )
    df["pq_largest_checking_or_savings_balance"] = df[
        [
            "plaid_largest_checking_or_savings_balance",
            "quovo_largest_checking_or_savings_balance",
        ]
    ].max(axis=1)
    df["pq_available_bal"] = df[["plaid_available_bal", "quovo_available_bal"]].max(
        axis=1
    )

    X = df[features].values
    y = df[target_col_name].values

    pos_wgt_scaling_factor = ~y.sum() / y.sum()
    lgb_default_params["pos_wgt_scaling_factor"] = pos_wgt_scaling_factor

    clf = lgb.LGBMClassifier(**lgb_default_params)

    fsel = Boruta(clf, drop_at=-15, max_iter=50, random_state=10, thresh=0.3, verbose=1)

    dimp = select_features(X, y, fsel)

    # look for high corr features
    if "corr_thresh" in CONFIG_FILE:
        corr_thresh = CONFIG_FILE["corr_thresh"]
    else:
        corr_thresh = 0.7

    dimp = dimp.set_index("feature", drop=True)
    corr = df[features].corr()

    passes_corr_check = []

    for f in features:
        passes = True

        for f2, cv in corr[f].iteritems():
            if f == f2:
                continue

            if cv > corr_thresh:
                if dimp.loc[f]["score"] < dimp.loc[f2]["score"]:
                    passes = False
                elif dimp.loc[f]["score"] == dimp.loc[f2]["score"]:
                    if dimp.loc[f]["mean_imp"] < dimp.loc[f2]["mean_imp"]:
                        passes = False

        passes_corr_check.append(passes)

    dimp["passes_corr_check"] = passes_corr_check
    dimp = dimp.reset_index()

    timestamp_str = str(int(time.time()))
    dimp.to_csv(
        os.path.join(src_base, f"../artifacts/fsel_res_{timestamp_str}.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
