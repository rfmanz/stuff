import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')


def plot_score_dist(dev_score, prod_score, figsize=(9,7), 
                    dev_label='development', prod_label='production',
                    bins=10, title=''):
    """ Compare distributions of two scores
    @params dev_score: sequence of scores
    @params prod_score: sequence of scores
    @params figsize: tuple
    @returns fig: plt.figure object
    """
    fig = plt.figure(figsize=figsize)
    plt.hist(dev_score, alpha=0.5, bins=bins, label=dev_label)
    plt.hist(prod_score, alpha=0.5, bins=bins, label=prod_label)
    plt.legend()
    plt.title(title)
    return fig


def get_psi(expected_df, actual_df, buckettype='bins', 
            buckets=10, axis=0, sort_psi=True):
    """ Wrapper function for rdsutils.validation.psi """
    from rdsutils.psi import calculate_psi
    assert(expected_df.columns.equals(actual_df.columns))
    
    features = expected_df.columns.tolist()
    psis = []
    for f in features:
        psis.append(calculate_psi(expected_df[f], actual_df[f], 
                                  buckettype=buckettype, buckets=buckets,
                                  axis=axis))
    psi_df = pd.DataFrame({'feature': features,
                           'psi': psis})
    if sort_psi:
        psi_df.sort_values(by=['psi'], ascending=False, inplace=True)
    return psi_df


def get_overlap_and_diff(s1, s2):
    """
    Get shared and different values in s1 and s2
    
    @params s1: sequences 1
    @params s2: sequences 2
    @returns result: dict
        'only s1': values uniquely in s1
        'only s2': values uniquely in s2
        'shared': shared values
    """
    result = {}
    val1 = set(s1)
    val2 = set(s2)
    result['only s1'] = val1 - val2
    result['only s2'] = val2 - val1
    result['shared'] = val1.intersection(val2)
    
    for k, v in result.items():
        result[k] = sorted(list(v))
    return result


def plot_difference(expected_sequence, actual_sequence, 
                    figsize=(9,7), bins=10, title=''):
    """ Plot the distr of difference of two sequences 
    Two sequences should have the same size
    """
    assert(len(expected_sequence) == len(actual_sequence))
    fig = plt.figure(figsize=figsize)
    seq = np.array(expected_sequence) - np.array(actual_sequence)
    plt.hist(seq, alpha=0.6, bins=bins, label='exp - actual')
    plt.legend()
    plt.title(title)
    return fig


def get_combined_dfs(expected_df, actual_df, index_col, cols=None, how='inner'):
    """ Combine and align two dfs base on the index col 

    - dfs will be merged on index_col
    - features in each dataframe are suffixed with '_exp' or '_act'
    - note: index_col should be unique indicators for each df
    """
    exp_df = expected_df.copy() if cols is None else expected_df[cols]
    act_df = actual_df.copy() if cols is None else actual_df[cols]
    exp_df.columns = [f + '_exp' if f != index_col else f for f in exp_df.columns]
    act_df.columns = [f + '_act' if f != index_col else f for f in act_df.columns]
    combined_df = pd.merge(exp_df, act_df, 
                           left_on=index_col, right_on=index_col,
                           how=how)
    return combined_df


def get_feature_comparison_report(expected_df, actual_df, index_col, 
                                  cols=None, save_path=None, explorative=True,
                                  **kwargs):
    """ Wrapper function for pandas_profiling report
    
    Generate html report for feature distr in the two reports.
    Features from two dfs are showed together for ease of comparison.
    
    if cols is not None, produce report on this subset
    
    @params expected_df
    @params actual_df
    @params index_col
    """
    from pandas_profiling import ProfileReport
    
    if index_col not in cols: cols.append(index_col)
    exp_df = expected_df.copy() if cols is None else expected_df[cols]
    act_df = actual_df.copy() if cols is None else actual_df[cols]
    df = get_combined_dfs(exp_df, act_df, index_col=index_col, cols=cols)
    df = df[sorted([f for f in df.columns if f != index_col])]
    
    report = ProfileReport(df, explorative=explorative, **kwargs)
    if save_path is not None:
        report.to_file(save_path)
    return report
    