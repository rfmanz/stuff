import pandas as pd
import matplotlib.pyplot as plt
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
    title = title or 'Feature Importance (avg over folds)'
    ax.set_title(title)
    plt.tight_layout()

    return fig, ax


def display_corr_matrix(corr_mtx_df, figsize=(9,7), title=None,
                        cmap='coolwarm', **kwargs):
    
    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(corr_mtx_df, cmap=cmap)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=360)
    title = title or 'Correlation Matrix of Highly Correlated Features'
    plt.title(title)
    plt.tight_layout()
    
    return fig, ax