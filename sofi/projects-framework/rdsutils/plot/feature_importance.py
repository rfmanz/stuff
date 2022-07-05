import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def display_feature_importance(features, importances, max_n_features=20, 
                               figsize=(9, 7), title=None, **kwargs):
    """Display feature importance given features
    and their importance.
    
    Parameters
    ----------
    features: pd.Series, np.ndarray - shape[n]
        Features.
    importances : pd.Series, np.ndarray - shape [n]
        Importance of respective feature.
    max_n_features : int
        Number of features to display.
    figsize : tuple (width, height)
        Size of figure to be produced.
    title : str
        Title of the axis.
    kwargs : keyword arguments
        Any keyword argument that can be passed to sns.barplot.
        
    Returns
    -------
    fig:
        matplotlib figure
    ax:
        matplotlib axis

    Notes
    -----
    This function is a tragedy. 
    TODO: make this better.
    """
    feature_importance_df = pd.DataFrame({'feature':features,
                                          'importance':importances})
    fi_grouped_mean = feature_importance_df.groupby('feature').mean()
    cols = fi_grouped_mean.sort_values(by='importance', ascending=False)\
                                       [:max_n_features].index
    fi_df = feature_importance_df.loc[feature_importance_df['feature']\
                                      .isin(cols)]\
                                      .sort_values(by='importance', 
                                                   ascending=False)
    
    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(x="importance", y="feature", data=fi_df, **kwargs)
    ax.set_title('Feature Importance (avg over folds)')
    plt.tight_layout()

    return fig, ax
