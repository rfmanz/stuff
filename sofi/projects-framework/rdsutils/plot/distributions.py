import math 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_feature_over_time(dt, feature, freq='1d', aggs=['mean', 'median', 'count'], 
                           groups=None, num_aggs_per_line=4, suptitle=None, figsize=None):
    """Plot a feature over time.
    
    Parameters
    ----------
    dt: pd.Series, np.ndarray - shape[n]
        Datetime.
    feature : pd.Series, np.ndarray - shape [n]
        Feature to be plotted.
    freq : string
        Frequency over which the feature should be binned.
        In the format {number of unit}{unit abbrevation} - e.g. 1d for 1 day
    aggs: list
        Aggregates to be plotted.
        Examples - mean, median, count.
    groups: pd.Series, np.ndarray - shape [n]
        To plot aggs accross several groups include groups.
        
    Returns
    -------
    fig:
        matplotlib figure
    axs:
        matplotlib axes
    """
    df = pd.DataFrame({'dt':np.array(dt), 'feature':np.array(feature),
                       'groups':np.array(groups)})
    
    unique_groups = df['groups'].unique()
    
    num_rows = math.ceil(len(aggs) / num_aggs_per_line)
    if not figsize:
        figsize = (min(num_aggs_per_line, len(aggs)) * 6, num_rows * 6)
    fig, axs = plt.subplots(nrows=num_rows, ncols=min(num_aggs_per_line, 
                                                      len(aggs)), figsize=figsize)
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=22)
        
    grouper = pd.Grouper(key='dt', freq=freq)
    for group in unique_groups:
        if group is not None:
            df_grouped = df[df['groups'] == group].groupby(grouper)
        else:
            df_grouped = df.groupby(grouper)
            
        for ax, agg in zip(axs.flatten(), aggs):
            df_grouped['feature'].agg(agg).plot(ax=ax, label=group)
            ax.legend()
            ax.set_title(str(agg))

    return fig, axs




def distplot_by_group(a, groups, n_bins=None,  figsize=None, title=None, 
                      xlabel=None, ignore_errors=True, ax=None, fig=None,
                      **kwargs):
    """Plot seaborn distplot grouped by a categorical attribute.
    
    Parameters
    ----------
    a: pd.Series, np.ndarray - shape[n]
        Some attribute.
    groups : pd.Series, np.ndarray - shape [n]
        Group to which each element belongs to.
    n_bins : int
        Number of bins each histogram should have. If 
        None n_bins dynamically calculated with Freedman-
        Diaconis rule.
    figsize : tuple (width, height)
        Size of figure to be produced.
    title : str
        Title of the axis.
    xlabel : str
        x axis label of the axis.
    ignore_errors : bool
        If there is an error when plotting distplot for
        a group, skip and continue to next one.
    kwargs : keyword arguments
        Any keyword argument that can be passed to sns.distplot.
        
    Returns
    -------
    fig:
        matplotlib figure
    ax:
        matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    nans = np.isnan(a)
    a = a[~nans]
    groups = groups[~nans]
    
    if isinstance(groups, pd.Series):
        unique_groups = groups.unique()
    else:
        unique_groups = np.unique(groups)
    
    for group in unique_groups:
        try:
            x = a[groups == group]

            if n_bins is None:
                #Freedman-Diaconis rule for calculating number of bins
                iqr = np.subtract(*np.percentile(x, [75, 25]))
                h = 2 * iqr * (len(x)**(-1/3))
                nb = math.ceil((max(x) - min(x)) / (h * 3))
            else:
                nb = n_bins

            sns.distplot(a[groups == group], bins=nb, 
                         label=group, ax=ax, **kwargs)
        except:
            if not ignore_errors:
                print("Error in group {}.".format(str(group)))
                raise
        
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel(xlabel)

    return fig, ax


def hist_by_target(col, target_col, df, nbins=40, alpha=0.6, ax=None, figsize=None, title=None, density=True):
    """
    Plot a historgram showing the distribution of a column when target == 0 and target == 1.
    """
    if ax is None:
        plt.figure(figsize=figsize)
    
    for targ in df[target_col].unique():
        df[df[target_col] == targ][col].hist(bins=nbins, ax=ax, label=str(targ), alpha=alpha, density=density)
    
    if title is None:
        title = str(col) + ' hist by target'
        if density:
            title += ' (density)'
        
    if ax is None:
        plt.title(title)
        plt.legend()
        plt.xlabel(col)
    else:
        ax.set_title(title)
        ax.legend()
        ax.set_xlabel(col)
        
        
        




