import json, os, sys, io, boto3
import pandas as pd


##################################################
#           Saving data statistics
##################################################

def get_data_statistics(config, modeling_df, 
                        valid_dfs=None, test_dfs=None, 
                        target_col="target", date_col="transaction_datetime",
                        output_json_path=None):
    """
    get df.shape, date_col info, target_col counts
    """
    
    stats = {}
    
    modeling_dfs = {"modeling_df": modeling_df}
    
    dfs = {}
    for dfs_ in [modeling_dfs, valid_dfs, test_dfs]:
        if dfs_ is None:
            continue
        for fname, df_ in dfs_.items():
            dfs[fname] = df_
     
    for fname, df in dfs.items():
        stats[fname] = {}
        if target_col and target_col in df.columns:
            stats[fname][target_col] = df[target_col].value_counts()
        else:
            print(f"{fname}: no {target_col}")
        if date_col and date_col in df.columns:
            stats[fname][date_col] = df[date_col].describe()
        else:
            print(f"{fname}: no {date_col}")
        
        print(f"produced statistics for {fname}")
        
    if output_json_path:
        import json
        stats_ = {}
        for fname, meta in stats.items():
            stats_[fname] = {}
            for col, df in meta.items():
                stats_[fname][col] = df.astype(str).to_dict()
                
        with open(output_json_path, "w") as f:
            json.dump(stats_, f, indent=4)
    return stats
    

##################################################
#           Saving model related info
##################################################

import matplotlib.pyplot as plt
from rdsutils.plot import plot_auc_curve_mult, plot_pr_curve_mult, plot_feature_over_time

plt.style.use('seaborn')


def save_hyperparams(params_dict, path, output_type="json"):

    if output_type == "json":
        dev_params = params_dict
        with open(path, "w") as f:
            json.dump(dev_params, f, indent=4)
    
    elif output_type == "csv":
        dev_params = {}
        for k, v in params_dict.items():
            dev_params[k] = str(v)
        params_df = pd.Series(dev_params).to_frame().reset_index()
        params_df.columns = ["hyper-parameter", "value"]
        params_df.to_csv(path, index=False)
    
    else:
        raise NotImplemented("Currently support json and csv format")
        
        
def save_feature_importance_plot(feature_name, feature_importances, path):
    import rdsutils.plot as rdsplot
    
    fig, ax = rdsplot.display_feature_importance(feature_name,
                                                 feature_importances,
                                                 max_n_features=-1)
    fig.savefig(path)

    

def get_binary_metrics(y_true, y_pred):
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scikitplot.helpers import binary_ks_curve
    
    auc = round(roc_auc_score(y_true=y_true,
                              y_score=y_pred)*100, 2)
    ap = round(average_precision_score(y_true=y_true,
                                       y_score=y_pred)*100, 2)
    _, _, _, ks, _, _ = binary_ks_curve(y_true=y_true, y_probas=y_pred)
    ks = round(ks*100, 2) 
    
    metrics = {'auc': auc,
               'ap': ap,
               'ks': ks}

    return metrics


def get_pred_reports(df, target_col, pred_cols):
    import pandas as pd
    result = {}
    for col in pred_cols:
        metrics = get_binary_metrics(df[target_col], df[col])
        result[col] = metrics
    return pd.DataFrame(result).T


def get_valid_performances(dfs, target_col, pred_cols):
    metrics = {}
    for fname, df in dfs.items():
        report = get_pred_reports(df, target_col, pred_cols)[["auc", "ap"]]
        metrics[fname] = report.sort_values("ap")
        
    return metrics

def save_valid_performances(dfs, target_col, pred_cols, dir_path):
    """
    save validation results to the directory. 
    each df has a "df_name.csv" file
    """
    import os, pandas as pd
    
    os.makedirs(dir_path, exist_ok=True)
    
    metrics = {}
    for fname, df in dfs.items():
        report = get_pred_reports(df, target_col, pred_cols)[["auc", "ap"]]
        metrics[fname] = report.sort_values("ap")
        report.to_csv(os.path.join(dir_path, f"{fname}.csv"))
        
    return metrics

def save_valid_performance_plots(dfs, target_col, pred_cols, dir_path):
    """
    save validation results to the directory. 
    each df has a "df_name.csv" file
    """
    import os, pandas as pd
    
    os.makedirs(dir_path, exist_ok=True)
    
    for fname, valid_df in dfs.items():
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
        

def save_lgbm_txt(model, path):
    import lightgbm as lgb
    assert(isinstance(model, lgb.LGBMClassifier))
    
    model.booster_.save_model(path)
    
def pickle_obj(obj, path):
    import pickle as pkl
    
    with open(path, "wb") as f:
        pkl.dump(obj, f)

# get feature order by importance

def get_feature_by_importance(model):
    fimp = sorted(zip(model.feature_importances_, model.feature_name_), reverse=True)
    features_by_imp = list(list(zip(*fimp))[1])    
    return features_by_imp


##################################################
#              SHAP
##################################################


# SHAP

import math, shap
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_shap_dependence(df, shap_values, features, ncols=6, figsize=None, **kwargs):
    """
    Build the partial dependence plot for a set of models and features.
    """
    nrows = math.ceil(len(features) / ncols)

    if figsize is None:
        figsize = (ncols * 6, nrows * 6)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for feature, ax in zip(features, axs.flatten()):
        shap.dependence_plot(feature, shap_values, df, 
                             ax=ax, show=False, **kwargs)
        rlim = df[feature].quantile(0.98)
        llim = df[feature].quantile(0.02) #- ((rlim - df[feature].quantile(0.02)) / 12)
            
        if rlim < np.inf and llim > -np.inf:
            ax.set_xlim(left=llim, right=rlim)
        
    return fig

def save_shap_dependence(model, modeling_df, features, dir_path):
    
    os.makedirs(dir_path, exist_ok=True)
    
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(modeling_df[model.feature_name_])
    
    pickle_obj(shap_values, os.path.join(dir_path, "shap_values.pkl"))
    
    fig = get_shap_dependence(modeling_df[model.feature_name_], 
                              shap_values[1], features, 
                              interaction_index=None)
    
    fig.savefig(os.path.join(dir_path, "shap_dependence.png"))
    
    
##################################################
#              PDP
##################################################

# PDP

import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def partial_dependency(model, df, feature, features, 
                       n_steps=10, sample_size=None):
    """
    Calculate partial dependency of a feature given a model.
    """
    if sample_size:
        d = df.sample(sample_size).copy()
    else:
        d = df.copy()
    grid = np.linspace(df[feature].quantile(0.001),
                       df[feature].quantile(.995),
                       n_steps)
    preds = []

    for x in grid:
        d[feature] = x
        y = np.average(model.predict(d[features]))
        preds.append([x, y])
    return np.array(preds).T[0], np.array(preds).T[1]


def partial_dependency_plot_cv(ax, models, df, 
                               feature, features, 
                               features_plot_order, n_steps=10, 
                               sample_size=None, ylab=None):
    """
    Return partial dependence plot for a feature on a set of models.
    """
    d = df.copy()

    partial_dependencies = []
    y_mean = np.array([0] * n_steps)
    x_mean = []

    y_min = np.inf
    y_max = -np.inf
    d[d[feature] == np.inf] = np.nan #edge case

    for model in models:
        x, y = partial_dependency(model, d, feature, 
                                  features, n_steps=n_steps, 
                                  sample_size=sample_size)
        
        y_min = min(y_min, min(y))
        y_max = max(y_max, max(y))
        
        y_mean = y_mean + (y / len(models))
        x_mean = x
        partial_dependencies.append([x, y])

        for x, y in partial_dependencies:
            ax.plot(x, y, '-', linewidth=1.4, alpha=0.6)

    ax.plot(x_mean, y_mean, '-', color = 'red', linewidth = 2.5)
    ax.set_xlim(d[feature].quantile(0.001), d[feature].quantile(0.995))
    ax.set_ylim(y_min*0.99, y_max*1.01)
    ax.set_xlabel(feature, fontsize = 10)
    if ylab:
        ax.set_ylabel(ylab, fontsize = 12)
                
            
def get_pdp(df, features, models, 
            features_plot_order, ncols=6, 
            figsize=None, sample_size=None):
    """
    Build the partial dependence plot for a set of models and features.
    """
    if type(models) is not list:
        models = [models]

    nrows = math.ceil(len(features) / ncols)

    if figsize is None:
        figsize = (ncols * 6, nrows * 6)

    fig, axs = plt.subplots(nrows=nrows, 
                            ncols=ncols, 
                            figsize=figsize)
    for feature, ax in tqdm(zip(features_plot_order, axs.flatten())):
        try:
            partial_dependency_plot_cv(ax, models, df, 
                                       feature, features, features_plot_order,  
                                       sample_size=sample_size)
        except:
            continue
    return fig


def save_pdp(model, modeling_df, features, 
             features_plot_order, dir_path):
    
    os.makedirs(dir_path, exist_ok=True)
    fig = get_pdp(modeling_df[features], features, model,
                  features_plot_order, ncols=6)
    fig.savefig(os.path.join(dir_path, "pdp.png"))
    

##################################################
#              WOE
##################################################

# WOE

import numpy as np
from rdsutils.woe import WOE_Transform
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_woe(fname, woe_dict, ax=None):
    if ax is None:
        fig = plt.figure()
    x = woe_dict[fname]["min"]
    y = woe_dict[fname]["woe"]
    ax.plot(x, y)
    ax.set_title(f"{fname}")
    

def get_woe_plots(df, woe_dict, features, ncols=6, figsize=None, **kwargs):
    """
    Build the partial dependence plot for a set of models and features.
    """
    nrows = math.ceil(len(features) / ncols)

    if figsize is None:
        figsize = (ncols * 6, nrows * 6)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for feature, ax in zip(features, axs.flatten()):
        plot_woe(feature, woe_dict, ax=ax)        
    return fig


def save_woe_plots(modeling_df, target_col, features, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    
    # fit woe
    woe = WOE_Transform(min_iv=-np.inf)
    woe.fit(modeling_df[features], modeling_df[target_col].astype(int), display=-1,)
    df = woe.transform(modeling_df[features], train_data=1, keep=False)

    # merge transformed data and record features
    modeling_df = modeling_df.merge(df, how='inner', left_index=True, right_index=True)
    features_woe = modeling_df.columns[modeling_df.columns.str.contains("woe")]

    # plot
    woe_dict = woe.woe_dict()
    woe_dict = dict([(k+"_woe", v) for k,v in woe_dict.items()])
    
    fig = get_woe_plots(modeling_df, woe_dict, features_woe)
    fig.savefig(os.path.join(dir_path, "woe_plot.png"))
    
    
##################################################
#              Segmented Performance
##################################################


from rdsutils.plot import plot_pr_curve
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import classification_report, confusion_matrix
import gc

def build_table1(modeling_df, percentiles, target_col_name, pred_col_name):
    """ cumulative """
    df = []
    for thresh, pctl in [(np.percentile(modeling_df[pred_col_name], pctl), pctl) for pctl in percentiles]:
        pred = modeling_df[pred_col_name] >= thresh
        rep = classification_report(y_true=modeling_df[target_col_name], y_pred=pred, output_dict=True)
        conf = confusion_matrix(y_true=modeling_df[target_col_name], y_pred=pred)
        tn, fp, fn, tp = conf.ravel()
        df.append([pctl, thresh, fp / (fp + tn) * 100, rep['True']['recall'] * 100,
                  sum(conf[:, 1]), conf[1][1], conf[1][0], conf[0][1], tn])
    return pd.DataFrame(df, columns=['Percentile', 'Threshold', 'False Positive Rate (%)', 
                                     'Fraud Capture Rate (%)', '#Above Threshold', '#Fraudulent Above Threshold', 
                                     '#Fraudulent Below Threshold', '#Good Above Threshold', '#True Negatives'])


def get_segmented_performances(valid_dfs, target_col, pred_cols, dir_path, model_name="final_model"):
    
    import os, pandas as pd
    
    os.makedirs(dir_path, exist_ok=True)
    
    for fname, valid_df in valid_dfs.items():
        for pred_col in pred_cols:
            if pred_col in ["pred", "score"]:
                mname = model_name
            else:
                mname = pred_col
            
            table = build_table1(valid_df, np.linspace(90, 99, 10)[::-1], target_col, pred_col)
            table.to_csv(os.path.join(dir_path, f"{mname}_top_10_pct.csv"), index=False)
            
            table = build_table1(valid_df, np.linspace(0, 90, 10)[::-1], target_col, pred_col)
            table.to_csv(os.path.join(dir_path, f"{mname}_decile.csv"), index=False)
    

##################################################
#              Output Data with Preds
################################################## 
    
def save_final_dfs(config, modeling_df, valid_dfs={}, test_dfs={}):
    
    def build_output_path(p):
        dir_path, fname = os.path.split(p)
        fname = "".join(["SCORED_"] + fname.split(".")[:-1] + [".parquet"])
        p = os.path.join(dir_path, fname)
        return p
    
    path = build_output_path(config["modeling_data_path"])
    print(f"modeling_df output to: {path}")
    modeling_df.to_parquet(path)
    
    for fname, df in valid_dfs.items():
        path = config["validation_data_paths"][fname]
        path = build_output_path(path)
        print(f"{fname} output to: {path}")
        df.to_parquet(path)
        
    for fname, df in test_dfs.items():
        path = config["inference_data_paths"][fname]
        path = build_output_path(path)
        print(f"{fname} output to: {path}")
        df.to_parquet(path)
    
    

##################################################
#              Put things together
##################################################

def drop_non_ach(df):
    df = df[df['transaction_code'].isin(['ACHDD']) & (df['transaction_amount'] > 0)]
    return df

def save_data_waterfall(modeling_df, dir_path):
    from rdsutils.data_waterfall import DataWaterfallReport 

    fns = [drop_non_ach,]
    wf = DataWaterfallReport(modeling_df, fns, 'is_returned')
    report = wf.get_report(True)
    report.to_csv(os.path.join(dir_path, "data_waterfall.csv"))
    
    
def save_governance_data(config, context, preprocess_fn):
    
    # set variables
    modeling_df = context["modeling_df"]
    valid_dfs = context["valid_dfs"]
    test_dfs = context["test_dfs"]
    pred_cols = context["pred_cols"]
    target_col = context["target_col"]
    date_col = context["date_col"]
    govn_path = context["govn_path"]
    model_name = context["model_name"]
    clf = context["model_object"]
    os.makedirs(govn_path, exist_ok=True)
    
    pred_cols = [col for col in pred_cols if col != "pred"]  # use score here
    pred_cols = list(set(pred_cols))
    
    # prep data
    print("preprocessing modeling_df")
    modeling_df = preprocess_fn(modeling_df.copy())
    
    # data waterfall
    print("saving data waterfall")
    save_data_waterfall(modeling_df, govn_path)
    
    # save hyperparam
    print("saving hyperparams")
    save_hyperparams(config["model_params"], os.path.join(govn_path, "model_params.json"), "json")
    save_hyperparams(config["model_params"], os.path.join(govn_path, "model_params.csv"), "csv")
    
    # save feature importance
    print("saving feature importance")
    save_feature_importance_plot(clf.feature_name_, clf.feature_importances_, 
                                 os.path.join(govn_path, "feature_importance.png"))
    
    # save valid performance 
    print("saving model validation performance")
    dir_path = os.path.join(govn_path, "oot_validation")
    save_valid_performances(valid_dfs, target_col, pred_cols, dir_path)
    save_valid_performance_plots(valid_dfs, target_col, pred_cols, dir_path)
    
    # save data and stats
    print("saving final dfs and statistics")
    get_data_statistics(config, modeling_df, valid_dfs, test_dfs,
                        target_col=target_col, date_col=date_col,
                        output_json_path=os.path.join(govn_path, "data_statistics.json"))
    save_final_dfs(config, modeling_df, valid_dfs, test_dfs)
    
    # model and model.txt
    print("saving model object")
    save_lgbm_txt(clf, os.path.join(govn_path, "model.txt"))
    pickle_obj(clf, os.path.join(govn_path, "model.pkl"))
    
    # feature importance order
    print("saving feature importance")
    features_by_imp = get_feature_by_importance(clf)
    
    # SHAP
    print("saving SHAP")
    save_shap_dependence(clf, modeling_df, features_by_imp,
                         os.path.join(govn_path, "shap"))
    
    # PDP
    print("saving PDP")
    save_pdp(clf, modeling_df, clf.feature_name_,
             features_by_imp, os.path.join(govn_path, "pdp"))
    
    # WOE
    print("saving WOE")
    save_woe_plots(modeling_df, target_col, 
                   features_by_imp, os.path.join(govn_path, "woe"))
    
    # segmented perfomrance
    print("saving validation segmented performance")
    get_segmented_performances(valid_dfs, target_col, pred_cols, 
                               os.path.join(govn_path, "valid_segmented_performance"),
                               model_name=model_name)