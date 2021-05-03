import pandas as pd
import numpy as np
import sys

from matplotlib import ticker
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as ticker


# import dtale
# dtale.show(df)

# from pandas_profiling import ProfileReport
# profile = ProfileReport(tr, explorative=True)
def peek(df, rows=3):
    return df.iloc[:rows, :].T


def describe_df(df):
    # Numerical
    print("--" * 20)
    print('Columns:', df.shape[1])
    print('Rows:', df.shape[0])
    print("Memory usage:", (f"({(sys.getsizeof(df) / 1024 ** 2):.2f} Mb)"))

    print("--" * 20)
    print('NUMERICAL VARIABLES:')

    numerical = df.select_dtypes(include=np.number)
    concatenated_numerical = pd.concat([
        round(numerical.isnull().sum() / df.shape[0] * 100, 2).astype(str) + "%",
        numerical.isnull().sum(),
        numerical.count(),
        numerical.min(),
        numerical.mean(),
        numerical.max()
    ], axis=1, keys=["%NULLS", "COUNT_NULLS", "NOT_NULL", 'MIN', 'MEAN', 'MAX'], sort=False).sort_values('COUNT_NULLS',
                                                                                                         ascending=False).reset_index().rename(
        columns={'index': ''})

    t = numerical.mode().T
    t.rename(columns={0: 'MODE'}, inplace=True)
    concatenated_numerical = concatenated_numerical.merge(t, how='left', left_on='', right_on=t.index)
    concatenated_numerical.index = concatenated_numerical.index + 1
    concatenated_numerical = concatenated_numerical.iloc[:, [0, 4, 5, 6, 7, 1, 2, 3]]

    print(tabulate(concatenated_numerical, headers=
    [
        'MIN',
        'MEAN',
        'MAX',
        'MODE',
        "%NULLS",
        "#_NULLS",
        "NOT_NULL",

    ], tablefmt="presto", colalign=("right"), floatfmt='.3f'))

    # Categorical

    print('-----' * 20)
    print()
    print('CATEGORICAL VARIABLES:')
    categorical = df.select_dtypes('object')
    if categorical.shape[1] == 0:
        print("No Categorical Variables")
    else:
        concatenated_categorical = pd.concat([

            round(categorical.isnull().sum() / df.shape[0] * 100, 2).astype(str) + "%",
            categorical.isnull().sum(),
            categorical.count()
        ],

            keys=["%NULLS",
                  "COUNT_NULLS",
                  "NOT_NULL"], axis=1, sort=False).sort_values('%NULLS', ascending=False).reset_index().rename(
            columns={'index': ''})

        max_unique = 5
        u_strs = []

        for col in categorical:
            series = categorical.loc[categorical[col].notnull(), col]
            n_unique = series.nunique()
            if n_unique > max_unique:
                u_strs.append(str(n_unique) + ' unique values')
            else:
                u_strs.append(str(series.unique()))

        t = pd.DataFrame(u_strs, categorical.columns)
        t = t.reset_index()
        t = t.rename(columns={'index': '', 0: 'Unique_Values'})
        concatenated_categorical = concatenated_categorical.merge(t, on='')
        concatenated_categorical.index = concatenated_categorical.index + 1

        print(tabulate(concatenated_categorical, headers=
        [
            "%NULLS",
            "#_NULLS",
            "NOT_NULL",
            "Unique_Values"

        ], tablefmt="presto", colalign=("left")))


def find_pretty_grid(n_plots, max_cols=5):
    """Determine a good grid shape for subplots.

    Tries to find a way to arange n_plots many subplots on a grid in a way
    that fills as many grid-cells as possible, while keeping the number
    of rows low and the number of columns below max_cols.

    Parameters
    ----------
    n_plots : int
        Number of plots to arrange.
    max_cols : int, default=5
        Maximum number of columns.

    Returns
    -------
    n_rows : int
        Number of rows in grid.
    n_cols : int
        Number of columns in grid.

    Examples
    --------
    >>> find_pretty_grid(16, 5)
    (4, 4)
    >>> find_pretty_grid(11, 5)
    (3, 4)
    >>> find_pretty_grid(10, 5)
    (2, 5)
    """

    # we could probably do something with prime numbers here
    # but looks like that becomes a combinatorial problem again?
    if n_plots % max_cols == 0:
        # perfect fit!
        # if max_cols is 6 do we prefer 6x1 over 3x2?
        return int(n_plots / max_cols), max_cols
    # min number of rows needed
    min_rows = int(np.ceil(n_plots / max_cols))
    best_empty = max_cols
    best_cols = max_cols
    for cols in range(max_cols, min_rows - 1, -1):
        # we only allow getting narrower if we have more cols than rows
        remainder = (n_plots % cols)
        empty = cols - remainder if remainder != 0 else 0
        if empty == 0:
            return int(n_plots / cols), cols
        if empty < best_empty:
            best_empty = empty
            best_cols = cols
    return int(np.ceil(n_plots / best_cols)), best_cols


def make_subplots(n_plots, max_cols=5, row_height=3):
    """Create a harmonious subplot grid.
    """
    n_rows, n_cols = find_pretty_grid(n_plots, max_cols=max_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, row_height * n_rows),
                             constrained_layout=True)
    # we don't want ravel to fail, this is awkward!
    axes = np.atleast_2d(axes)
    return fig, axes


def class_hists(data, column, target, bins="auto", ax=None, legend=False,
                scale_separately=True):
    """Grouped univariate histograms.

    Parameters
    ----------
    data : pandas DataFrame
        Input data to plot
    column : column specifier
        Column in the data to compute histograms over (must be continuous).
    target : column specifier
        Target column in data, must be categorical.
    bins : string, int or array-like
        Number of bins, 'auto' or bin edges. Passed to np.histogram_bin_edges.
        We always show at least 5 bins for now.
    ax : matplotlib axes
        Axes to plot into
    legend : boolean, default=False
        Whether to create a legend.
    scale_separately : boolean, default=True
        Whether to scale each class separately.

    Examples
    --------
    >>> from dabl.datasets import load_adult
    >>> data = load_adult()
    >>> class_hists(data, "age", "gender", legend=True)
    <matplotlib...
    """
    col_data = data[column].dropna()

    if ax is None:
        ax = plt.gca()
    if col_data.nunique() > 10:
        ordinal = False
        # histograms
        bin_edges = np.histogram_bin_edges(col_data, bins=bins)
        if len(bin_edges) > 30:
            bin_edges = np.histogram_bin_edges(col_data, bins=30)

        counts = {}
        for name, group in data.groupby(target)[column]:
            this_counts, _ = np.histogram(group, bins=bin_edges)
            counts[name] = this_counts
        counts = pd.DataFrame(counts)
    else:
        ordinal = True
        # ordinal data, count distinct values
        counts = data.groupby(target)[column].value_counts().unstack(target)
    if scale_separately:
        # normalize by maximum
        counts = counts / counts.max()
    bottom = counts.max().max() * 1.1
    for i, name in enumerate(counts.columns):
        if ordinal:
            ax.bar(range(counts.shape[0]), counts[name], width=.9,
                   bottom=bottom * i, tick_label=counts.index, linewidth=2,
                   edgecolor='k')
            xmin, xmax = 0 - .5, counts.shape[0] - .5
        else:
            ax.bar(bin_edges[:-1], counts[name], bottom=bottom * i, label=name,
                   align='edge', width=(bin_edges[1] - bin_edges[0]) * .9)
            xmin, xmax = bin_edges[0], bin_edges[-1]
        ax.hlines(bottom * i, xmin=xmin, xmax=xmax,
                  linewidth=1)
    if legend:
        ax.legend()
    ax.set_yticks(())
    ax.set_xlabel(column)
    return ax


def plot_univariate_classification(df, target_name):
    df[[target_name]] = df[[target_name]].astype('object')
    continuous_cols = list(df.select_dtypes("number").columns)
    fig, axes = make_subplots(n_plots=len(continuous_cols), row_height=2)
    for i, (ind, ax) in enumerate(zip(continuous_cols, axes.ravel())):
        class_hists(df, continuous_cols[i],
                    target_name, ax=ax, legend=i == 0)
    for j in range(i + 1, axes.size):
        # turn off axis if we didn't fill last row
        axes.ravel()[j].set_axis_off()
    return plt.show()


def violin_plot(df, target_name):
    continuous_cols = list(df.select_dtypes("number").columns)
    data = pd.DataFrame(StandardScaler().fit_transform(df[continuous_cols]), columns=df[continuous_cols].columns,
                        index=df[continuous_cols].index)
    data = pd.concat([data, df[[target_name]]], axis=1)
    data = pd.melt(data, id_vars=target_name,
                   var_name="features",
                   value_name='value')

    plt.figure(figsize=(10, 10))
    ax = sns.violinplot(x="features", y="value", hue=target_name, data=data, split=True, inner="quartile")
    for i in range(len(np.unique(data["features"])) - 1):
        ax.axvline(i + 0.5, color='grey', lw=1)
    plt.xticks(rotation=20)
    return plt.show()


def box_plot(df, target_name):
    continuous_cols = list(df.select_dtypes("number").columns)
    data = pd.DataFrame(StandardScaler().fit_transform(df[continuous_cols]), columns=df[continuous_cols].columns,
                        index=df[continuous_cols].index)
    data = pd.concat([data, df[[target_name]]], axis=1)
    data = pd.melt(data, id_vars=target_name,
                   var_name="features",
                   value_name='value')

    plt.figure(figsize=(10, 10))
    ax = sns.boxplot(x="features", y="value", hue="churn", data=data)
    for i in range(len(np.unique(data["features"])) - 1):
        ax.axvline(i + 0.5, color='grey', lw=1)
    plt.xticks(rotation=20)
    return plt.show()


def plot_density_numerical_for_whole_dataframe(df):
    continuous_cols = list(df.select_dtypes("number").columns)
    fig, axes = make_subplots(n_plots=len(continuous_cols), row_height=2)
    for i, (ind, ax) in enumerate(zip(continuous_cols, axes.ravel())):
        sns.kdeplot(df[continuous_cols[i]], color='black', shade=True, legend=True, ax=ax)
    for j in range(i + 1, axes.size):
        # turn off axis if we didn't fill last row
        axes.ravel()[j].set_axis_off()
    return plt.show()


def plot_single_numerical(df):
    plt.figure()
    sns.kdeplot(df, color="black", shade="gray")
    return plt.show()


def ordered_barplot(df, variable_name):
    s = df[~pd.isnull(df[[variable_name]])][variable_name]
    chart = pd.value_counts(s).to_frame(name='data')
    chart.index.name = 'labels'
    chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x="labels", y="data", data=chart)
    ncount = len(df[[variable_name]])
    # plt.title('Distribution of Truck Configurations')
    plt.xlabel(variable_name)

    # Make twin axis
    ax2 = ax.twinx()

    # Switch so count axis is on right, frequency on left
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()

    # Also switch the labels over
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')

    ax2.set_ylabel('Frequency [%]')

    for p in ax.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate('{:.1f}%'.format(100. * y / ncount), (x.mean(), y),
                    ha='center', va='bottom')  # set the alignment of the text

    # Use a LinearLocator to ensure the correct number of ticks
    ax.yaxis.set_major_locator(ticker.LinearLocator(11))

    # Fix the frequency range to 0-100
    ax2.set_ylim(0, 100)
    ax.set_ylim(0, ncount)

    # And use a MultipleLocator to ensure a tick spacing of 10
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

    # Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
    #ax2.grid(None)
    return plt.show()
