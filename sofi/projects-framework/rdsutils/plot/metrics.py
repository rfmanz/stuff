import math 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

################################################
#                  AP PLOTS
################################################


def plot_pr_curve(y_score, y_true, figsize=None):
    """Plot a precision-recall curve.
    
    Parameters
    ----------
    y_score: pd.Series, np.ndarray - shape[n]
        Scores/predictions.
    y_true : pd.Series, np.ndarray - shape [n]
        Targets.
        
    Returns
    -------
    fig:
        matplotlib figure
    ax:
        matplotlib axis
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    average_precision = average_precision_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP = {}'\
              .format(str(average_precision)))
    
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




################################################
#                  AUC PLOTS
################################################

    
def plot_auc_curve(y_score, y_true, figsize=None):
    """Plot a precision-recall curve.
    
    Parameters
    ----------
    y_score: pd.Series, np.ndarray - shape[n]
        Scores/predictions.
    y_true : pd.Series, np.ndarray - shape [n]
        Targets.
        
    Returns
    -------
    fig:
        matplotlib figure
    ax:
        matplotlib axis
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=figsize)
    
    aucroc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    plt.step(fpr, tpr, color='b', alpha=0.2,
             where='post')
    plt.fill_between(fpr, tpr, alpha=0.2, color='b')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class AUC ROC curve: AUC = {}'\
              .format(str(aucroc)))
    
    return fig, ax


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

