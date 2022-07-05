"""
metrics.py
"""

import numpy as np
import pandas as pd


def compare_classification(y_true, y_score, metrics=[], score_names=None):
    """Compare several scores accross various metrics.
    
    Parameters
    ----------
    y_score : pd.DataFrame, np.ndarray - shape[n, m]
        Array of scores. n = number of examples, m = number of scores.
    y_true : int/string
        True labels.
    metrics : list of tuples
        Metrics used in evaluation.
        Format for each metric is (name, metric, kwargs) where
        name is a string, metric is a function that takes y_score,
        y_true, and kwargs as input and optional kwargs. kwargs is an 
        optional dictionary but is options.
        
    Returns
    -------
    df : pandas DataFrame
    """
    raise NotImplementedError
    if isinstance(pd.DataFrame, y_score) and not score_names:
        score_names = y_score.columns
        y_score = y_score.values
    
#     df = []
#     for metric in metrics:
#         row = []
#         _, fn, kwargs = _get_metric(metric)
        
    

def get_quantile_table(y_true, y_score, metrics=[], lower_bound=0, upper_bound=100, num_steps=10, score_name=None):
    """Get quantile table with various metrics.
    
    Parameters
    ----------
    y_score : pd.Series, np.ndarray - shape[n]
        Array of scores. n = number of examples.
    y_true : int/string
        True labels.
    metrics : list of tuples
        Metrics used in evaluation.
        Format for each metric is (name, metric, kwargs) where
        name is a string, metric is a function that takes y_score,
        y_true, and kwargs as input and optional kwargs. kwargs is an 
        optional dictionary but is options.
        
    Returns
    -------
    df : pandas DataFrame
    """        
    percentiles = np.linspace(lower_bound, upper_bound, 
                              num_steps, endpoint=False)
    
    # remove null values (not handlded by np.percentile)
    nans = np.isnan(y_score)
    y_score = y_score[~nans]
    y_true = y_true[~nans]
    
    df = []
    for thresh, pctl in [(np.percentile(y_score, pctl), 
                          pctl) for pctl in percentiles]:
        row = [pctl, thresh]
        y_pred = y_score >= thresh
        
        for metric in metrics:
            _, fn, kwargs = _get_metric(metric)
            row.append(fn(y_true, y_pred, **kwargs))
            
        df.append(row)
    
    cnames = [m[0] for m in metrics]
    return pd.DataFrame(df, columns=['Percentile', 'Threshold']+cnames)


def _get_metric(metric):
    """Utility function used by compare_classification, 
    get_quantile_table.
    """
    if len(metric) == 3:
        return metric[0], metric[1], metric[2]
    else:
        return metric[0], metric[1], {}


def get_pred_reports(df, target_col, pred_cols, sample_weight_col=None, dropna=True):
    import pandas as pd

    result = {}
    for col in pred_cols:
        if dropna:
            df_ = df[~df[col].isna()]
        
        if sample_weight_col is None:
            metrics = get_binary_metrics(df_[target_col], 
                                         df_[col])
        if sample_weight_col is not None:
            metrics = get_binary_metrics(df_[target_col], 
                                         df_[col], df_[sample_weight_col])
        result[col] = metrics
    return pd.DataFrame(result).T
    
    
    
def get_binary_metrics(y_true, y_pred, sample_weight=None, 
                       round_by=4, nr_quantiles=100):
    """
    y_pred follows the sklearn tradition of higher leads to higher likelihood of positive class
    
    in most cases
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scikitplot.helpers import binary_ks_curve
    
    auc = round(roc_auc_score(y_true=y_true,
                              y_score=y_pred,
                              sample_weight=sample_weight), round_by)
    ap = round(average_precision_score(y_true=y_true,
                                       y_score=y_pred,
                                       sample_weight=sample_weight), round_by)
    ks = score_gain(y_true, y_pred, sample_weight=sample_weight,
                    nr_quantiles=nr_quantiles, direction=1,
                    round_by=round_by)['KS'].max()
    
    metrics = {'auc': auc,
               'ap': ap,
               'ks': ks}

    return metrics


def score_gain(y_true, y_pred, sample_weight=None, nr_quantiles=10, 
               direction=0, round_by=6):
    """
    @params y_true
    @params y_pred
    @params sample_weight
    """
    if sample_weight is None:
        sample_weight = pd.Series(np.ones(len(y_true)))
    Y, Y_weight, X = y_true, sample_weight, y_pred
    YY=Y.reset_index(drop=True)
    YY_weight=Y_weight.reset_index(drop=True)
    XX=X.reset_index(drop=True)
    
    bins=pd.qcut(XX,nr_quantiles,retbins=False,labels=False,duplicates="drop")
    Y_bins=pd.DataFrame(bins,index=YY.index)
    Y_model=pd.DataFrame(XX,index=YY.index)
    Y_new=YY*YY_weight
        
    Y_final=pd.concat([Y_new,YY_weight,Y_model,Y_bins],axis=1) 
    Y_final.columns=['true','weight','score','bins']
    Y_final.loc[Y_final.score==0,'bins']=-99     
    s_min=round(Y_final.groupby('bins')['score'].min(),round_by)
    s_max=round(Y_final.groupby('bins')['score'].max(),round_by)
    s_count=Y_final.groupby('bins')['weight'].sum()
    s_pred=round(Y_final.groupby('bins')['score'].mean(),round_by)
    #s_rate=round(Y_final.groupby('bins')['true'].mean(),4)
    s_numtarget=round(Y_final.groupby('bins')['true'].sum(),1)
    s_rate=s_numtarget/s_count
    g_table=pd.concat([s_count,s_min,s_max,s_pred,s_rate,s_numtarget],axis=1)    
    g_table.columns=["#accts","min","max","pred_rate","actual_rate","target_num"]
        
    if direction==1:
        g_table=g_table.sort_values(by="min",ascending=False)
    elif direction==0:
        g_table=g_table.sort_values(by="min",ascending=True)
        
    g_table['nontarget_num']=g_table['#accts']-g_table['target_num']
        
    g_table['pct_cum_acct']=round(g_table['#accts'].cumsum()/g_table['#accts'].sum(),round_by)
    g_table['pct_cum_target']=round(g_table['target_num'].cumsum()/g_table['target_num'].sum(),round_by)
    g_table['pct_cum_nontarget']=round(g_table['nontarget_num'].cumsum()/g_table['nontarget_num'].sum(),round_by)
    g_table['reverse_target_num']=g_table.target_num.values[::-1]
    g_table['reverse_total_num']=g_table['#accts'].values[::-1]
    g_table['cum_bads']=round(g_table['reverse_target_num'].cumsum()/g_table['reverse_total_num'].cumsum(),round_by)
    g_table['cum_acct']=round(g_table['reverse_total_num'].cumsum()/g_table['reverse_total_num'].sum(),round_by)
    g_table['KS']=(g_table['pct_cum_target'] - g_table['pct_cum_nontarget'])*100
        
    g_table.reset_index(inplace=True,drop=True)
    g_table=g_table[["pct_cum_acct","#accts","min","max","target_num",'pct_cum_target','pct_cum_nontarget',"actual_rate",'KS','cum_bads','cum_acct']]
        
    return g_table


