import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_swapset_table(df, seg1, seg2, margins=True):
    table = df[[seg1, seg2]].value_counts(
        normalize=False).sort_index().reset_index()
    table.columns = [seg1, seg2, 'counts']
    table = pd.pivot_table(table, values='counts', index=seg1, 
                           columns=seg2, aggfunc='sum', 
                           fill_value=0, margins=margins)
    return table.astype(int)

def get_target_counts_table(df, seg1, seg2, target_col, margins=True):
    table = df[[target_col, seg1, seg2]].groupby([seg1, seg2])[target_col].sum().sort_index().reset_index()
    table.columns = [seg1, seg2, 'counts']
    table = pd.pivot_table(table, values='counts', index=seg1, 
                           columns=seg2, aggfunc='sum', 
                           fill_value=0, margins=margins)
    return table.astype(int)

def get_swap_set_bad_rate(df: pd.DataFrame, seg1:str, seg2:str, target_col:str, margins:bool=True):
    """
    get swap set bad rate
    
    @params df: dataframe containing columns - seg1, seg2, target_col
    @params seg1: incumbent model segmentation column
    @params seg2: new model segmentation column
    @params target_col: binary target column [Bool/Int]
    @params margins: compute margins
    """
    assert(df[target_col].nunique() == 2)
    rg_tbl = get_swapset_table(df, seg1, seg2, margins=margins)
    tgt_tbl = get_target_counts_table(df, seg1, seg2, target_col, margins=margins)
    return tgt_tbl/rg_tbl

def plot_swap_set_bad_rate(df, seg1, seg2, target_col, margins=True, 
                           fig=None, ax=None, title=None,
                           xlabel=None, ylabel=None, cmap="coolwarm"):
    """
    @params df: dataframe containing columns - seg1, seg2, target_col
    @params seg1: incumbent model segmentation column
    @params seg2: new model segmentation column
    @params target_col: binary target column [Bool/Int]
    @params margins: compute margins
    
    other params are shared across seaborn/matplotlib.pyplot
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    swapset = get_swap_set_bad_rate(df, seg1, seg2, target_col, margins=margins)
    swapset_pct = swapset * 100
    ax = sns.heatmap(swapset_pct, cmap=cmap, annot=True, fmt='.2f', ax=ax)
    for t in ax.texts: t.set_text(t.get_text() + " %")
        
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

    return fig, ax