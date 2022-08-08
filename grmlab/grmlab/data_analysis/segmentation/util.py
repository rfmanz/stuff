"""
Data segmentation utils.
"""

# Authors: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2020.

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from scipy.stats import f
from scipy.stats import norm

from ...data_processing.preprocessing import Preprocessing
from ...modelling.feature_selection import SelectOptimize


RANDOM_INT = 100000


def snappy_model(x_values, y_values, target, numeric_variables,
                 max_correlation=0.6, n_max_features=None,
                 max_correlation_target=0.3, excluded=None, fixed=None):

    x_values = scale_sample(x_values, numeric_variables)

    # preprocessing to eliminate constants and nans columns
    preprocessing = Preprocessing(verbose=False)
    preprocessing.run(x_values)
    preprocessing.transform(data=x_values, mode="basic")

    if n_max_features is None:
        n_max_features = x_values.shape[1]

    if fixed is not None:
        fixed = list(set(x_values.columns).intersection(fixed))

    if excluded is not None:
        excluded = list(set(x_values.columns).intersection(excluded))

    if x_values.shape[1] > 1:
        optimize_all = SelectOptimize(
            feature_names=x_values.columns.values, method="classification",
            max_correlation=max_correlation, n_max_features=n_max_features,
            max_correlation_target=max_correlation_target,
            excluded=excluded, fixed=fixed)
        optimize_all.fit(x_values.values, y_values[target].values)
        vars_all = x_values.columns[optimize_all.support_]
    else:
        vars_all = x_values.columns

    log_fit_all = LogisticRegression().fit(
        x_values[vars_all].values, y_values[target].values)

    y_pred = log_fit_all.predict_proba(x_values[vars_all].values)

    fpr, tpr, thresholds = roc_curve(y_values[target].values,
                                             [elt[1] for elt in y_pred])

    feat_importance = linear_feature_importance(
        coef=log_fit_all.coef_.ravel(), X_trans=x_values[vars_all].values)

    gini = 2*auc(fpr, tpr) - 1

    return {"gini": gini, "fpr": fpr, "tpr": tpr, "y_pred": y_pred,
            "variables": vars_all, "model": log_fit_all,
            "feat_importance": feat_importance}


def linear_feature_importance(coef, X_trans):
    """
    Feature importance approximation.

    This approximation is exact for normal regression.
    """
    std_features = np.std(X_trans, axis=0)
    weights = np.abs(coef) * std_features
    return weights / weights.sum()


def scale_sample(data, numeric_variables):

    if len(numeric_variables) > 0:
        X_trans = StandardScaler().fit_transform(data[numeric_variables])
        X_trans = pd.DataFrame(X_trans, columns=numeric_variables)

        return X_trans.merge(data.drop(columns=numeric_variables),
                             left_index=True, right_index=True)
    else:
        return data


def allison_test(X, y, mask1, mask2, target, numeric_variables,
                 max_correlation=0.6, max_correlation_target=0.3,
                 type_reg="linear", excluded=None, fixed=None, verbose=True):

    # check type_reg
    if isinstance(type_reg, str):
        if not ((type_reg == "linear") or (type_reg == "logit")):
            raise ValueError("type_reg options are 'linear' or 'logit', not "
                             "{}".format(type_reg))
    else:
        raise TypeError("type_reg must be class str not "
                        "{}.".format(type(type_reg)))

    variables = X.columns
    X_corr = X[variables]

    X_s0_v1 = X_corr[mask1]
    X_s1_v1 = X_corr[mask2]

    X_v1 = np.concatenate((X_s0_v1, X_s1_v1))
    X_v3 = np.concatenate((X_s0_v1, X_s1_v1*0))

    X_allison = np.concatenate((X_v1, X_v3), axis=1)

    y_allison = np.concatenate((y[mask1],
                                y[mask2])).ravel()

    df_allison = pd.DataFrame(
        X_allison,
        columns=([elt+"_v1" for elt in X_corr.columns] +
                 [elt+"_v3" for elt in X_corr.columns]))

    # preprocessing to eliminate constants and nans columns
    preprocessing = Preprocessing(verbose=False)
    preprocessing.run(df_allison)
    preprocessing.transform(data=df_allison, mode="basic")
    numeric_variables = list(map(str.lower, numeric_variables))

    if fixed is not None:
        fixed = list(set(X_corr.columns).intersection(fixed))
        fixed = [elt+"_v1" for elt in fixed] + [elt+"_v3" for elt in fixed]
        fixed = list(set(df_allison.columns.values).intersection(fixed))
    else:
        fixed = []
    if excluded is not None:
        excluded = list(set(X_corr.columns).intersection(excluded))
        excluded = [elt+"_v1" for elt in excluded] + [
            elt+"_v3" for elt in excluded]
        excluded = list(set(df_allison.columns.values).intersection(excluded))

    df_allison["dummy_segment"] = np.concatenate(
        (np.full(X_s0_v1.shape[0], 1), np.full(X_s1_v1.shape[0], 0)))

    if df_allison.shape[1] > 1:
        optimize2 = SelectOptimize(
            feature_names=df_allison.columns.values, method="classification",
            max_correlation=max_correlation, fixed=["dummy_segment"]+fixed,
            excluded=excluded, n_min_features=None,
            n_max_features=df_allison.shape[1],
            max_correlation_target=max_correlation_target)
        optimize2.fit(df_allison.values, y_allison)
        final_vars = df_allison.columns[optimize2.support_]
    else:
        final_vars = df_allison.columns

    df_allison = df_allison[final_vars]

    num_vars = np.array(
        [[elt + "_v1", elt+"_v3"] for elt in numeric_variables]).ravel()
    num_vars_df = final_vars[final_vars.isin(num_vars)]
    df_allison = scale_sample(df_allison, num_vars_df)

    df_allison = sm.add_constant(df_allison)

    # regression
    if type_reg == "linear":
        reg_all = sm.OLS(y_allison, df_allison)
        fit_reg_all = reg_all.fit()
    elif type_reg == "logit":
        reg_all = sm.Logit(y_allison, df_allison)
        fit_reg_all = reg_all.fit(maxiter=100, method='lbfgs')

    if verbose:
        print(fit_reg_all.summary())

    p_features = fit_reg_all.pvalues[fit_reg_all.pvalues < 0.05]

    if verbose:
        print(p_features)

    if p_features.shape[0] == 0:
        dict_groups = {"group_1": np.array([]), "dummy": np.array([]),
                       "group_3": np.array([])}
    else:
        dict_groups = {
            "group_1": p_features.index[
                p_features.index.str.contains("_v1")].values,
            "dummy": p_features.index[
                p_features.index == "dummy_segment"].values,
            "group_3": p_features.index[
                p_features.index.str.contains("_v3")].values}

    return dict_groups, fit_reg_all


def chow_test(X1, y1, X2, y2, alpha=0.05):

    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, y2))

    k = X.shape[1] + 1

    reg_all = LinearRegression().fit(X, y)
    SSR_all = np.sum(np.power(y - reg_all.predict(X), 2))

    reg_right = LinearRegression().fit(X1, y1)
    N_right = X1.shape[0]
    SSR_right = np.sum(np.power(y1 - reg_right.predict(X1), 2))

    reg_left = LinearRegression().fit(X2, y2)
    N_left = X2.shape[0]
    SSR_left = np.sum(np.power(y2 - reg_left.predict(X2), 2))

    # test statistics
    F_statistic = (((SSR_all - (SSR_right + SSR_left))/k) /
                   ((SSR_right + SSR_left)/(N_right+N_left-2*k)))
    p_value = 1 - f.cdf(F_statistic, k, (N_right+N_left-2*k))
    F_critical = f.ppf(1-alpha, k, (N_right+N_left-2*k), loc=0, scale=1)

    if F_statistic < F_critical:
        return False, p_value
    elif F_statistic >= F_critical:
        return True, p_value


def odds_test(y1, y2, alpha=0.05):
    # H0: segment category and target category are independent

    # n11 == segment 1 with y=1
    n11 = y1.sum()
    n10 = y1.shape[0] - n11

    n21 = y2.sum()
    n20 = y2.shape[0] - n21

    if n10*n21 == 0:
        OR = ((n11+0.5)*(n20+0.5))/((n10+0.5)*(n21+0.5))
    else:
        OR = (n11*n20)/(n10*n21)

    if 0 in (n11, n10, n21, n20):
        V = 1/(n11+0.5) + 1/(n10+0.5) + 1/(n21+0.5) + 1/(n20+0.5)
    else:
        V = 1/n11 + 1/n10 + 1/n21 + 1/n20

    T_statistic = np.log(OR)/np.sqrt(V)
    p_value = 1 - norm.cdf(T_statistic)

    T_critical = norm.ppf(1-alpha)

    if T_statistic < T_critical:
        return False, p_value
    elif T_statistic >= T_critical:
        return True, p_value


def calc_iv(leafs, y):
    n_malos = y.values.sum()
    n_buenos = y.shape[0] - n_malos
    iv = 0
    for leaf in leafs:
        malos = y[leaf.mask].values.sum()
        buenos = y[leaf.mask].shape[0] - malos

        perc_malos = malos/n_malos
        perc_buenos = buenos/n_buenos

        if perc_malos == 0 or perc_buenos == 0:
            iv_bucket = 0
        else:
            iv_bucket = (perc_buenos-perc_malos)*np.log(perc_buenos/perc_malos)

        iv += iv_bucket
    return iv
