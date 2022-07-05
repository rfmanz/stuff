import pandas as pd


def get_feature_info_unsup_(series):
    """Get feature info (unsupervised).
    
    Parameters
    ----------
    series : pd.Series
        feature column
    fns : dict
        Dictionary of functions to apply

    Returns
    -------
    stats : some stats
    """
    stats = {}
    if pd.api.types.is_numeric_dtype(series):
        stats["mean"] = series.mean()
        stats["std"] = series.std()
        stats["variance"] = series.var()
        stats["min"] = series.min()
        stats["max"] = series.max()
        stats["iqr"] = iqr(series)
        stats["kurtosis"] = series.kurt()
        stats["skewness"] = series.skew()
        stats["sum"] = series.sum()
        stats["mad"] = series.mad()
        stats["n_zeros"] = (len(series) - np.count_nonzero(series))
        stats["p_zeros"] = round(100 * float(stats["n_zeros"]) / len(series), 2)
        stats["cv"] = stats["std"] / stats["mean"] if stats["mean"] else np.NaN

    stats["n_unique"] = len(series.unique())
    stats["p_unique"] = round(100 * float(stats["n_unique"] / len(series)), 2)
    
    stats["n_missing"] = series.isna().sum()
    stats["p_missing"] = round(100 * float(stats["n_missing"] / len(series)), 2)
            
    return stats


def get_missing_(df, features, missing_threshold):
    data = df[features]
    
    # pct missing
    missing_serie = data.isna().sum(axis=0) / data.shape[0] 
        
    # produce df
    stats = missing_serie.to_frame().reset_index()
    stats.columns = ['feature', 'missing_frac']
    record_missing = stats[stats.missing_frac > missing_threshold]
    record_missing = record_missing.sort_values(by=['missing_frac'], 
                                                ascending=False)
    to_drop = record_missing.feature.tolist()
    
    return record_missing, to_drop


def get_single_unique_(df, features, count_nan):
    data = df[features]
    
    counts = data.nunique(axis=0)
    counts = counts.to_frame().reset_index()
    counts.columns = ['feature', 'nr_unique']
        
    if count_nan:
        nan_vals = data.isna().any(axis=0).astype(int).values
        counts['nr_unique'] += nan_vals
    
    record_single_unique = counts[counts.nr_unique==1]
    to_drop = record_single_unique.feature.tolist()
    
    return record_single_unique, to_drop