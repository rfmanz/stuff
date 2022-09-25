# reference 
# https://github.com/smazzanti/mrmr/blob/main/mrmr/main.py

from joblib import Parallel, delayed
from multiprocessing import cpu_count
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif as sklearn_f_classif
from sklearn.feature_selection import f_regression as sklearn_f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
import lightgbm as lgb

warnings.filterwarnings("ignore")
import tqdm

FLOOR = 0.00001

#####################################################################
# Functions for parallelization


def parallel_df(func, df, series, weights=None):
    n_jobs = min(cpu_count(), len(df.columns))
    col_chunks = np.array_split(range(len(df.columns)), n_jobs)
    lst = Parallel(n_jobs=n_jobs)(
        delayed(func)(df.iloc[:, col_chunk], series, weights) for col_chunk in col_chunks
    )
    return pd.concat(lst)


#####################################################################
# Functions for computing relevance and redundancy


def _f_classif(X, y, weights=None):
    if weights is not None:
        raise NotImplementedError("Not implemented")
        
    def _f_classif_series(x, y):
        x_not_na = ~x.isna()
        if x_not_na.sum() == 0:
            return 0
        return sklearn_f_classif(x[x_not_na].to_frame(), y[x_not_na])[0][0]

    return X.apply(lambda col: _f_classif_series(col, y)).fillna(0.0)


def _f_regression(X, y, weights=None):
    if weights is not None:
        raise NotImplementedError("Not implemented")
        
    def _f_regression_series(x, y):
        x_not_na = ~x.isna()
        if x_not_na.sum() == 0:
            return 0
        return sklearn_f_regression(x[x_not_na].to_frame(), y[x_not_na])[0][0]

    return X.apply(lambda col: _f_regression_series(col, y)).fillna(0.0)


def _corr_pearson(A, b, weights=None):
    if weights is None:
        corr = A.corrwith(b)
    else:
        corr = nanwcorr_with(A, b, weights)
    return corr.fillna(0.0).abs().clip(FLOOR)

def nanwcorr_with(df: pd.DataFrame, series: pd.Series, 
                  weights: pd.Series=None):
    """ compute nancorr with weighted options """
    import math
    
    if weights is None:
        return df.corrwith(series)
    
    assert df.shape[0] == len(series) == len(weights)
    
    X, y, w = df, series, weights
    result = []
    for col in X.columns:
        x = X[col]
        valid = (~x.isna()) & (~y.isna()) & (~w.isna())
        x_, y_, w_ = x[valid].values, y[valid].values, w[valid].values
        cor_ = _wcov(x_, y_, w_)/math.sqrt(_wcov(x_, x_, w_) * _wcov(y_, y_, w_))
        result.append(cor_)
    result = pd.Series(data=result, index=X.columns)
    return result


def _wcov(x, y, w):
    wtot = np.sum(w) + 0.00000001  # for not exploding
    mx, my = np.sum(x*w)/wtot, np.sum(y*w)/wtot    
    return np.sum(w*(x-mx)*(y-mx))/wtot


#####################################################################
# Functions for computing relevance and redundancy
# Parallelized versions (except random_forest_classif which cannot be parallelized)


def f_classif(X, y, weights=None):
    """Compute F-statistic between DataFrame X and Series y"""
    if weights is not None:
        raise NotImplementedError("Have not implemented weighted mrmr functionality. Please check out classif example")

    return parallel_df(_f_classif, X, y)


def f_regression(X, y, weights=None):
    """Compute F-statistic between DataFrame X and Series y"""
    if weights is not None:
        raise NotImplementedError("Have not implemented weighted mrmr functionality. Please check out classif example")

    return parallel_df(_f_regression, X, y)


def random_forest_classif(X, y, weights=None):
    """Compute feature importance of each column of DataFrame X after fitting a random forest on Series y"""
    if weights is not None:
        raise NotImplementedError("Have not implemented weighted mrmr functionality. Please check out classif example")

    forest = RandomForestClassifier(max_depth=5, random_state=0).fit(
        X.fillna(X.min().min() - 1), y, sample_weight=weights
    )
    return pd.Series(forest.feature_importances_, index=X.columns)

def random_forest_regression(X, y, weights=None):
    """Compute feature importance of each column of DataFrame X after fitting a random forest on Series y"""
    forest = RandomForestRegressor(max_depth=5, random_state=0).fit(
        X.fillna(X.min().min() - 1), y, sample_weights=weights
    )
    return pd.Series(forest.feature_importances_, index=X.columns)


def corr_pearson(A, b, weights=None):
    """Compute Pearson correlation between DataFrame A and Series b"""
    return parallel_df(_corr_pearson, A, b, weights)

def get_iv(X, y, weights=pd.Series([], dtype=float)):
    from .woe import WOE_Transform
    woe = WOE_Transform(min_iv=-np.inf)
    woe.fit(X, y.astype(int), Y_weight=weights, display=-1)
    iv_tbl = woe.get_iv()
    iv_tbl.set_index("attr", inplace=True)
    return iv_tbl.iv


#####################################################################


def encode_df(X, y, cat_features, cat_encoding):
    import category_encoders as ce

    ENCODERS = {
        "leave_one_out": ce.LeaveOneOutEncoder(
            cols=cat_features, handle_missing="return_nan"
        ),
        "james_stein": ce.JamesSteinEncoder(
            cols=cat_features, handle_missing="return_nan"
        ),
        "target": ce.TargetEncoder(cols=cat_features, handle_missing="return_nan"),
    }

    X = ENCODERS[cat_encoding].fit_transform(X, y)

    return X


def get_mrmr_rankings(mrmr_features, all_features):
    ranks = list(range(1, len(mrmr_features) + 1)) + [len(all_features)] * (len(all_features) - len(mrmr_features))
    idx = mrmr_features + list(set(ranking.index) - set(mrmr_features))
    return pd.Series(ranks, idx).sort_values(ascending=True)


#####################################################################
# MRMR selection


def _mrmr_base(
    X,
    y,
    K,
    weights,
    func_relevance,
    func_redundancy,
    func_denominator,
    cat_features=None,
    cat_encoding="leave_one_out",
    only_same_domain=False,
):
    """
    Do MRMR selection.

    Args:
        X: (pandas.DataFrame) A Dataframe consisting of numeric features only.
        y: (pandas.Series) A Series containing the target variable.
        K: (int) Number of features to select.
        weights: (pandas.Series) A Series containing the target weights
        func_relevance: (func) Relevance function.
        func_redundancy: (func) Redundancy function.
        func_denominator: (func) Synthesis function to apply to the denominator.
        cat_features: (list) List of categorical features. If None, all string columns will be encoded
        cat_encoding: (str) Name of categorical encoding. Supported: 'leave_one_out', 'james_stein', 'target'
        only_same_domain: (bool) If False, all the necessary correlation coefficients are computed.
            If True, only features belonging to the same domain are compared.
            Domain is defined by the string preceding the first underscore:
            for instance "cusinfo_age" and "cusinfo_income" belong to the same domain,whereas "age" and "income" don't.

    Returns:
        (list) List of K names of selected features (sorted by importance).
    """

    # encode categorical features
    X = encode_df(X, y, cat_features=cat_features, cat_encoding=cat_encoding)

    # compute relevance
    rel = func_relevance(X, y, weights)

    # keep only columns that have positive relevance
    columns = rel[rel.fillna(0) > 0].index.to_list()
    K = min(K, len(columns))
    rel = rel.loc[columns]

    # init
    red = pd.DataFrame(FLOOR, index=columns, columns=columns)
    selected = []
    not_selected = columns.copy()

    for i in tqdm.tqdm(range(K)):

        # compute score numerator
        score_numerator = rel.loc[not_selected]

        # compute score denominator
        if i > 0:

            last_selected = selected[-1]

            if only_same_domain:
                not_selected_subset = [
                    c
                    for c in not_selected
                    if c.split("_")[0] == last_selected.split("_")[0]
                ]
            else:
                not_selected_subset = not_selected

            if not_selected_subset:
                red.loc[not_selected_subset, last_selected] = (
                    func_redundancy(X[not_selected_subset], X[last_selected], weights)
                    .abs()
                    .clip(FLOOR)
                    .fillna(FLOOR)
                )
                score_denominator = (
                    red.loc[not_selected, selected]
                    .apply(func_denominator, axis=1)
                    .round(5)
                    .replace(1.0, float("Inf"))
                )

        else:
            score_denominator = pd.Series(1, index=columns)

        # compute score and select best
        score = score_numerator / score_denominator
        best = score.index[score.argmax()]
        selected.append(best)
        not_selected.remove(best)

    return selected


def mrmr_classif(
    X,
    y,
    K,
    weights=None,
    relevance="f",
    redundancy="c",
    denominator="mean",
    cat_features=None,
    cat_encoding="leave_one_out",
    only_same_domain=False,
):
    """
    Do MRMR feature selection on classification task.

    Args:
        X: (pandas.DataFrame) A Dataframe consisting of numeric features only.
        y: (pandas.Series) A Series containing the (categorical) target variable.
        K: (int) Number of features to select.
        weights: (pandas.Series) A Series containing the target weights
        relevance: (str or function) Relevance method.
            If function, it should take X and y as input and return a pandas.Series containing a (non-negative) score of relevance for each feature of X.
            If string, name of method, supported: 'f' (f-statistic), 'rf' (random forest).
        redundancy: (str or function) Redundancy method.
            If function, it should take A and b as input and return a pandas.Series containing a (non-negative) score of redundancy for each feature of A.
            If string, name of method, supported: 'c' (Pearson correlation)
        denominator: (str or function) Synthesis function to apply to the denominator of MRMR score.
            If function, it should take an iterable as input and return a scalar.
            If string, name of method, supported: 'max', 'mean'
        cat_features: (list) List of categorical features. If None, all string columns will be encoded
        cat_encoding: (str) Name of categorical encoding. Supported: 'leave_one_out', 'james_stein', 'target'
        only_same_domain: (bool) If False, all the necessary correlation coefficients are computed.
            If True, only features belonging to the same domain are compared.
            Domain is defined by the string preceding the first underscore:
            for instance "cusinfo_age" and "cusinfo_income" belong to the same domain,
            whereas "age" and "income" don't.

    Returns:
        (list) List of K names of selected features (sorted by importance).
    """
    
    FUNCS = {
        "f": f_classif,
        "rf": random_forest_classif,
        "c": corr_pearson,
        "mean": np.mean,
        "max": np.max,
        "iv": get_iv
    }

    func_relevance = FUNCS[relevance] if relevance in FUNCS.keys() else relevance
    func_redundancy = FUNCS[redundancy] if redundancy in FUNCS.keys() else redundancy
    func_denominator = (
        FUNCS[denominator] if denominator in FUNCS.keys() else denominator
    )

    return _mrmr_base(
        X,
        y,
        K,
        weights,
        func_relevance,
        func_redundancy,
        func_denominator,
        cat_features,
        cat_encoding,
        only_same_domain,
    )


def mrmr_regression(
    X,
    y,
    K,
    weights=None,
    relevance="f",
    redundancy="c",
    denominator="mean",
    cat_features=None,
    cat_encoding="leave_one_out",
    only_same_domain=False,
):
    """
    Do MRMR feature selection on regression task.

    Args:
        X: (pandas.DataFrame) A Dataframe consisting of numeric features only.
        y: (pandas.Series) A Series containing the (numerical) target variable.
        K: (int) Number of features to select.
        relevance: (str or function) Relevance method.
            If function, it should take X and y as input and return a pandas.Series containing a (non-negative) score of relevance for each feature of X.
            If string, name of method, supported: 'f' (f-statistic), 'rf' (random forest).
        redundancy: (str or function) Redundancy method.
            If function, it should take A and b as input and return a pandas.Series containing a (non-negative) score of redundancy for each feature of A.
            If string, name of method, supported: 'c' (Pearson correlation)
        denominator: (str or function) Synthesis function to apply to the denominator of MRMR score.
            If function, it should take an iterable as input and return a scalar.
            If string, name of method, supported: 'max', 'mean'
        cat_features: (list) List of categorical features. If None, all string columns will be encoded
        cat_encoding: (str) Name of categorical encoding. Supported: 'leave_one_out', 'james_stein', 'target'
        only_same_domain: (bool) If False, all the necessary correlation coefficients are computed.
            If True, only features belonging to the same domain are compared.
            Domain is defined by the string preceding the first underscore:
            for instance "cusinfo_age" and "cusinfo_income" belong to the same domain,
            whereas "age" and "income" don't.

    Returns:
        (list) List of K names of selected features (sorted by importance).
    """
    if weights is not None:
        raise NotImplementedError("Have not implemented weighted mrmr functionality. Please check out classif example")

    FUNCS = {
        "f": f_regression,
        "rf": random_forest_regression,
        "c": corr_pearson,
        "mean": np.mean,
        "max": np.max,
    }

    func_relevance = FUNCS[relevance] if relevance in FUNCS.keys() else relevance
    func_redundancy = FUNCS[redundancy] if redundancy in FUNCS.keys() else redundancy
    func_denominator = (
        FUNCS[denominator] if denominator in FUNCS.keys() else denominator
    )

    return _mrmr_base(
        X,
        y,
        K,
        weights,
        func_relevance,
        func_redundancy,
        func_denominator,
        cat_features,
        cat_encoding,
        only_same_domain,
    )


#####################################################################
# MRMR selection augmented by conditional 

def lgbm_classif(X, y, selected_fts, mc_dict=None):
    """
    X: features df
    y: target sereis
    select_fts: list of selected features
    mc_dict: used to obtain mc constraints
    """
    mc_cstr = None
    
    fts = list(set(X.columns) - set(selected_fts))
    
    # get enforced dir of previously selected features
    selected_mc = None
    if mc_dict is not None:
        selected_mc = [mc_dict[f] for f in selected_fts]
    
    fimps = []
    for f in fts:
        fts_ = selected_fts + [f]
        
        # i know...i m wordy..
        if mc_dict is not None and f not in mc_dict:
            fimps.append(-1)
            continue
        elif mc_dict is not None and f in mc_dict:
            mc_cstr = selected_mc + [mc_dict[f]]
        
        
        tree = lgb.LGBMClassifier(max_depth=5, random_state=42,
                                  monotone_constraints=mc_cstr)
        tree.fit(X[fts_], y)
        
        fimp_ = tree.feature_importances_[tree.feature_name_.index(f)]
        fimps.append(fimp_)
        
    return pd.Series(fimps, index=fts)


def cond_mrmr_classif(
    X,
    y,
    K,
    relevance="rf",
    redundancy="c",
    denominator="mean",
    compute_freq=1,
    cat_features=None,
    cat_encoding="leave_one_out",
    only_same_domain=False,
    mc_dict=None
):
    FUNCS = {
        "lgbm": lgbm_classif,
        "c": corr_pearson,
        "mean": np.mean,
        "max": np.max,
    }
    
    func_relevance = FUNCS[relevance] if relevance in FUNCS.keys() else relevance
    func_redundancy = FUNCS[redundancy] if redundancy in FUNCS.keys() else redundancy
    func_denominator = (
        FUNCS[denominator] if denominator in FUNCS.keys() else denominator
    )
    
    return _cond_mrmr_base(
        X,
        y,
        K,
        func_relevance,
        func_redundancy,
        func_denominator,
        compute_freq,
        cat_features,
        cat_encoding,
        only_same_domain,
        mc_dict
    )

def _cond_mrmr_base(
    X,
    y,
    K,
    func_relevance,
    func_redundancy,
    func_denominator,
    compute_freq=1,
    cat_features=None,
    cat_encoding="leave_one_out",
    only_same_domain=False,
    mc_dict=None
):
    """
    same as _mrmr_base, except for we re-compute relevance 
    conditioned on selected features per {compute_freq} features
    """
    # encode categorical features
    X = encode_df(X, y, cat_features=cat_features, cat_encoding=cat_encoding)

    # relevance
    selected = []
    rel = func_relevance(X, y, selected, mc_dict)  # experiment with no mc_constriants for now   
    # keep only columns that have positive relevance
    columns = rel[rel.fillna(0) > 0].index.to_list()
    K = min(K, len(columns))
    rel = rel.loc[columns]
    
    # init
    red = pd.DataFrame(FLOOR, index=columns, columns=columns)
    not_selected = columns.copy()

    counter = 0
    for i in tqdm.tqdm(range(K)):
        # compute score numerator
        if (i > 0) and (counter % compute_freq == 0):
            print(f"refetching relevance - counter: {counter}, n selected: {len(selected)}, n not_selected: {len(not_selected)}")
            rel = func_relevance(X, y, selected, mc_dict)  # experiment with no mc_constriants for now
            
            # keep only columns that have positive relevance
            columns = rel[rel.fillna(0) > 0].index.to_list()
            K = min(K, len(columns))
            rel = rel.loc[columns]
            not_selected = columns.copy()
            
        score_numerator = rel.loc[not_selected]

        # compute score denominator
        if i > 0:

            last_selected = selected[-1]

            if only_same_domain:
                not_selected_subset = [
                    c
                    for c in not_selected
                    if c.split("_")[0] == last_selected.split("_")[0]
                ]
            else:
                not_selected_subset = not_selected

            if not_selected_subset:
                red.loc[not_selected_subset, last_selected] = (
                    func_redundancy(X[not_selected_subset], X[last_selected])
                    .abs()
                    .clip(FLOOR)
                    .fillna(FLOOR)
                )
                score_denominator = (
                    red.loc[not_selected, selected]
                    .apply(func_denominator, axis=1)
                    .round(5)
                    .replace(1.0, float("Inf"))
                )

        else:
            score_denominator = pd.Series(1, index=columns)

        # compute score and select best
        score = score_numerator / score_denominator
        best = score.index[score.argmax()]
        selected.append(best)
        not_selected.remove(best)
        
        counter += 1

    return selected