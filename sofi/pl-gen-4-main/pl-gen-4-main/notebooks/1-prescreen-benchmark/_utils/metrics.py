# Weighted performance metrics
# pred AUC, KS

# Plot
# Gains table
# BAD Rate by Percentile
# ROC Curve
# Efficient Frontier

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
import scipy.stats.stats as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import seaborn as sns
# plt.style.use('ggplot')



def plot_auc_curve_mult(y_test, y_scores, sample_weight=None, title="", 
                        colors=['b'], figsize=None, fill_area=True,
                        alpha=0.4, ax=None, fig=None, **kwargs):
    """Plot multiple AUC-ROC curves on the same canvas.
    
    Parameters
    ----------
    y_test : pd.Series, np.ndarray - shape [n]
        Targets.
    y_scores: [(y_score, label), ...]
        list of tuples with y_score and corresponding label
        y_score - pd.Series
        label - str - label to show in legend
    sample_weight: pd.Series
    title: str
        title for the plot
    colors: list(str)
        list of colors for the y_scores respectively
    figsize: tuple(int)
        figure size for plt
    
        
    Returns
    -------
    fig:
        matplotlib figure
    ax:
        matplotlib axis
    """
    assert(len(y_scores) == len(colors))
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt
    
    aucs = []
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    for i in range(len(y_scores)):
        y_score, label = y_scores[i]
        
        aucroc = roc_auc_score(y_test, y_score,
                               sample_weight=sample_weight)
        aucs.append(aucroc)
        fpr, tpr, _ = roc_curve(y_test, y_score, 
                                sample_weight=sample_weight)

        ax.step(fpr, tpr, color=colors[i], alpha=alpha,
                 where='post', label=label)
        if fill_area:
            ax.fill_between(fpr, tpr, alpha=alpha, color=colors[i])
        
    
    plt.legend()

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(title)
    
    return fig, ax


def plot_pr_curve_mult(y_test, y_scores, sample_weight=None, 
                       title=None, colors=['b'], figsize=None,
                       fill_area=True, alpha=0.4,
                       ax=None, fig=None, **kwargs):
    """Plot multiple precision-recall curves on the same canvas.
    
    Parameters
    ----------
    y_test : pd.Series, np.ndarray - shape [n]
        Targets.
    y_scores: [(y_score, label), ...]
        list of tuples with y_score and corresponding label
        y_score - pd.Series
        label - str - label to show in legend
    sample_weight: pd.Series
    title: str
        title for the plot
    colors: list(str)
        list of colors for the y_scores respectively
    figsize: tuple(int)
        figure size for plt
    
        
    Returns
    -------
    fig:
        matplotlib figure
    ax:
        matplotlib axis
    """
    assert(len(y_scores) == len(colors))
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    
    aps = []
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
            
    for i in range(len(y_scores)):
        y_score, label = y_scores[i]
        
        average_precision = average_precision_score(y_test, y_score, 
                                                    sample_weight=sample_weight)
        aps.append(average_precision)
        precision, recall, _ = precision_recall_curve(y_test, y_score,
                                                      sample_weight=sample_weight,
                                                      **kwargs)

        ax.step(recall, precision, color=colors[i], alpha=alpha,
                 where='post', label=label)
        if fill_area:
            ax.fill_between(recall, precision, alpha=alpha, color=colors[i])
    
    ax.legend()

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(title)
    
    return fig, ax



def get_binary_metrics(y_true, y_pred, sample_weight=None, 
                       direction=1, round_by=4, nr_quantiles=100):
    """
    direction: {0 or 1}
        1: higher y_pred indicates higher likelihood of target class
        0: higher y_pred indicates lower likelihood of target class
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
                    nr_quantiles=nr_quantiles, direction=direction,
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


