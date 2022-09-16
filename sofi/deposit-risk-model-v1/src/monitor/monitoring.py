import sys, os
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from .utils import scale_scores, preprocess
from rdsutils.post_implementation_testing import (
    plot_score_dist,
    get_psi,
    get_overlap_and_diff,
)
from rdsutils.post_implementation_testing import (
    plot_difference,
    get_combined_dfs,
    get_feature_comparison_report,
)
from rdsutils.plot import (
    plot_auc_curve_mult,
    plot_pr_curve_mult,
    plot_feature_over_time,
)


####################################
#            Get Data
#            Get Data
####################################


def load_df(path, format="auto"):
    """ load dataframe from path """
    import pandas as pd

    if format == "parquet":
        return pd.read_parquet(path)
    elif format == "feather":
        return pd.read_feather(path)
    elif format == "csv":
        return pd.read_csv(path)
    elif format == "auto":
        format_ = path.split(".")[-1]
        return getattr(pd, f"read_{format_}")(path)
    else:
        raise ValueError("Unknown format")


####################################
#          Process Data
####################################


def process_dfs(df_dev, df_prod, df_modeling, context={}):
    """customize based on tasks
    @params df_dev: pd.DataFrame
        - development data at time of monitoring
    @params df_prod:
        - production data at time of monitoring
    @params df_modeling:
        - training data used to produce the model
    """
    # dev data
    df_dev.loc[:, "party_id"] = df_dev["user_id"]
    active_accounts = df_dev[
        (df_dev.nr_past_transactions > 0) & (df_dev.nr_transactions_30d > 0)
    ].party_id.unique()
    df_dev.loc[:, "is_active"] = df_dev.party_id.isin(active_accounts)

    # get data on the specific sample date
    # dev data can be sampled
    monitoring_date = context["monitoring_date"]  # or raise an error
    df_dev = df_dev[df_dev.sample_date == pd.to_datetime(monitoring_date)]
    df_dev = df_dev.drop_duplicates()
    df_dev = preprocess(df_dev)

    # prod data
    df_prod.loc[:, "is_active"] = df_prod.party_id.isin(active_accounts)
    df_prod = df_prod.drop_duplicates()
    df_prod = preprocess(df_prod)

    # join production data with targets
    combined_df = df_prod.merge(
        df_dev[["party_id", "target", "indeterminate", "fico_score"]],
        how="inner",
        on="party_id",
    )
    # flip the sign to get positive corr with riskiness
    combined_df.loc[:, "fico_adjusted"] = (combined_df["fico_score"] 
                                           * np.where(combined_df["fico_score"] > 850, 0, 1))
    combined_df.loc[:, "fico_adjusted_pred"] = -combined_df["fico_adjusted"]

    # modeling_df - original training data
    #last_sample_date = context["last_sample_date"]
    # modeling_df_snapshot = df_modeling[df_modeling.sample_date == last_sample_date]
    modeling_df_snapshot = df_modeling.loc[df_modeling.groupby("borrower_id").sample_date.idxmax()]
    modeling_df_snapshot = modeling_df_snapshot.drop_duplicates()
    modeling_df_snapshot = preprocess(modeling_df_snapshot)
    # join training data with party_id
    modeling_df_snapshot = modeling_df_snapshot.merge(
        df_dev[["borrower_id", "party_id"]], how="left", on="borrower_id"
    )
    # output
    context["combined_df"] = combined_df[combined_df.is_active]
    context["df_dev"] = df_dev[df_dev.is_active]
    context["df_prod"] = df_prod[df_prod.is_active]
    context["df_train"] = modeling_df_snapshot

    return context


def verify_dev_data(df_dev):
    """ check validity of dev data if needed """
    raise NotImplemented


def verify_prod_data(df_prod):
    """check validity of prod data if needed

    for example:
    - check the score by prod_data is the same
      as if we evaluate again with offline model.
    - features are already preprocessed
    """
    raise NotImplemented


####################################
#            Metrics
####################################


def get_binary_metrics(y_true, y_pred):
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scikitplot.helpers import binary_ks_curve

    auc = round(roc_auc_score(y_true=y_true, y_score=y_pred) * 100, 2)
    ap = round(average_precision_score(y_true=y_true, y_score=y_pred) * 100, 2)
    _, _, _, ks, _, _ = binary_ks_curve(y_true=y_true, y_probas=y_pred)
    ks = round(ks * 100, 2)

    metrics = {"auc": auc, "ap": ap, "ks": ks}

    return metrics


def get_pred_reports(df, target_col, pred_cols, dropna=True):
    import pandas as pd

    result = {}
    for col in pred_cols:
        if dropna:
            df_ = df[~df[col].isna()]
        metrics = get_binary_metrics(df_[target_col], df_[col])
        result[col] = metrics
    return pd.DataFrame(result).T


####################################
#         Feature Reports
####################################


def get_feature_report(df_dev, df_prod, df_train, report_path, cols):
    """"""
    from rdsutils.post_implementation_testing import get_feature_comparison_report

    # get borrower_id, party_id mapping
    modeling_df = df_train.merge(
        df_dev[["borrower_id", "party_id"]], how="left", on="borrower_id"
    )
    report = get_feature_comparison_report(
        df_train, df_prod, "party_id", title="post-imp-report", cols=cols
    )
    report.to_file(report_path)


####################################
#             Plots
####################################


def save_valid_performance_plots(dfs, target_col, pred_cols, dir_path):
    """
    save validation results to the directory.
    each df has a "df_name.csv" file
    """
    import os, pandas as pd

    os.makedirs(dir_path, exist_ok=True)

    for fname, valid_df in dfs.items():
        preds = [(valid_df[col], col) for col in pred_cols]

        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        cmap = plt.get_cmap("Set1")
        colors = [cmap(i) for i in range(len(preds))]  # np.linspace(0, 1, len(preds))]

        title = "Precision-Recall curve: Baseline Comparison"
        plot_pr_curve_mult(
            valid_df[target_col], preds, title=title, colors=colors, ax=axs[0], fig=fig
        )

        title = "AUC-ROC curve: Baseline Comparison"
        plot_auc_curve_mult(
            valid_df[target_col], preds, title=title, colors=colors, ax=axs[1], fig=fig
        )

        fig.savefig(os.path.join(dir_path, f"{fname}_auc_ap.png"))


####################################
#       Segmented performances
####################################

from rdsutils.plot import plot_pr_curve
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import classification_report, confusion_matrix
import gc


def build_table1(modeling_df, percentiles, target_col_name, pred_col_name):
    """ cumulative """
    df = []
    for thresh, pctl in [
        (np.percentile(modeling_df[pred_col_name], pctl), pctl) for pctl in percentiles
    ]:
        pred = modeling_df[pred_col_name] >= thresh
        rep = classification_report(
            y_true=modeling_df[target_col_name], y_pred=pred, output_dict=True
        )
        conf = confusion_matrix(y_true=modeling_df[target_col_name], y_pred=pred)
        tn, fp, fn, tp = conf.ravel()
        df.append(
            [
                pctl,
                thresh,
                fp / (fp + tn) * 100,
                rep["True"]["recall"] * 100,
                sum(conf[:, 1]),
                conf[1][1],
                conf[1][0],
                conf[0][1],
                tn,
            ]
        )
    return pd.DataFrame(
        df,
        columns=[
            "Percentile",
            "Threshold",
            "False Positive Rate (%)",
            "Fraud Capture Rate (%)",
            "#Above Threshold",
            "#Fraudulent Above Threshold",
            "#Fraudulent Below Threshold",
            "#Good Above Threshold",
            "#True Negatives",
        ],
    )


def get_segmented_performances(
    valid_dfs, target_col, pred_cols, dir_path, model_name="final_model"
):

    import os, pandas as pd

    os.makedirs(dir_path, exist_ok=True)

    for fname, valid_df in valid_dfs.items():
        for pred_col in pred_cols:
            if pred_col in ["pred", "score"]:
                mname = model_name
            else:
                mname = pred_col

            table = build_table1(
                valid_df, np.linspace(90, 99, 10)[::-1], target_col, pred_col
            )
            table.round(2).to_csv(os.path.join(dir_path, f"{mname}_top_10_pct.csv"), index=False)

            table = build_table1(
                valid_df, np.linspace(0, 90, 10)[::-1], target_col, pred_col
            )
            table.round(2).to_csv(os.path.join(dir_path, f"{mname}_decile.csv"), index=False)

            
####################################
#          Model vs Fico
####################################

def build_table_3(modeling_df, percentiles, target_col_name, pred_col_name, fico_col_name):
    """ Thomas's table 3... model vs fico """
    df = []
    for (fthresh, pctl), (mthresh, pctl) in zip([(np.percentile(modeling_df[fico_col_name], pctl), pctl) for pctl in percentiles], [(np.percentile(modeling_df[pred_col_name], pctl), pctl) for pctl in percentiles][::-1]):
        mbr = modeling_df[modeling_df[pred_col_name] >= mthresh]['target'].mean() * 100
        fbr = modeling_df[modeling_df[fico_col_name] <= fthresh]['target'].mean() * 100
        
        m_ct = len(modeling_df[modeling_df[pred_col_name] >= mthresh])
        f_ct = len(modeling_df[modeling_df[fico_col_name] <= fthresh])
        df.append([pctl, fthresh, mthresh, fbr, mbr, (f_ct, m_ct)])
        
    return pd.DataFrame(df, columns=['Percentile', 'FICO Threshold', 'Model Score Treshold', 'FICO Bad Rate (%)', 'Model Bad Rate (%)', 'Nr records above Threshold (Fico, Model)'])
 
    
def get_model_vs_baseline(
    df, target_col, model_col, baseline_col, dir_path, percentiles=np.linspace(0, 100, 26)
):
    """
    get segmented model vs baseline performances
    and save to directory
    
    Please make sure df[[target_col, model_col, baseline_col]] doesn't have Nan
    """
    import os, pandas as pd
    os.makedirs(dir_path, exist_ok=True)

    table = build_table_3(df, percentiles, target_col,
                          model_col, baseline_col)
    table.round(2).to_csv(os.path.join(dir_path, "model-vs-fico.csv"), index=False)
    
    
    
####################################
#         Performance Plots
####################################


def save_valid_performance_plots(dfs, target_col, pred_cols, dir_path, dropna=False):
    """
    save validation results to the directory. 
    each df has a "df_name.csv" file
    """
    import os, pandas as pd
    
    os.makedirs(dir_path, exist_ok=True)
    
    for fname, valid_df in dfs.items():
        if dropna:
            valid_df = valid_df.copy()
            valid_df = valid_df[~valid_df[pred_cols].isna().any(axis=1)]
            
        preds = [(valid_df[col], col) for col in pred_cols]

        fig, axs = plt.subplots(1, 2, figsize=(16,6))
        cmap = plt.get_cmap('Set1')
        colors = [cmap(i) for i in range(len(preds))]# np.linspace(0, 1, len(preds))]

        title = 'Precision-Recall curve: Baseline Comparison'
        plot_pr_curve_mult(valid_df[target_col], preds,
                           title=title, colors=colors, 
                           ax=axs[0], fig=fig) 

        title = 'AUC-ROC curve: Baseline Comparison'
        plot_auc_curve_mult(valid_df[target_col], preds,
                           title=title, colors=colors, 
                           ax=axs[1], fig=fig)

        fig.savefig(os.path.join(dir_path, f"{fname}_auc_ap.png"))