import numpy as np
import pandas as pd
import lightgbm as lgb


def partial_dependency_(
    model: lgb.Booster,
    df: pd.DataFrame,
    feature: str,
    features: list,
    num_cuts=20,
    include_missing=True,
    missing_idx=-1,
):
    """
    providing a fixed feature, compute partial dependency by value quantiles

    @params model: lgb.Booster or model object with a predict function
    @params df: pd.DataFrame - dataset for pdp computation
    @params feature: str - specific feature to look at
    @params features: list[str] - feature list for model prediction
    @params num_cuts: int - number of value cutoffs/quantiles
    @params include_missing: bool - whether to include a row indicate missing values
        if missing is included, the row has id = missing_idx  and median = NaN
    @params missing_idx: int - which index to encode missing

    @return df_cutoff: pd.DataFrame with the following columns
        bins, min, max, grid(median), pred
        if missing exist, append a row with attr_label == NaN
    """
    # create cutoffs
    qtls = list(np.linspace(0.001, 0.999, num_cuts))
    cutoffs = [df[feature].quantile(x) for x in qtls]
    cutoffs = cutoffs + [-np.inf, np.inf]
    cutoffs = list(set(cutoffs))
    cutoffs.sort()

    # build bins
    df_ = df[features].copy()
    bins = list(range(1, len(cutoffs)))
    df_["bins"] = pd.cut(
        df_[feature], cutoffs, right=True, labels=bins, include_lowest=True
    )

    # build df table
    df_cutoff = pd.DataFrame()
    grouper_obj = df_[~df_[feature].isna()].groupby("bins")[feature]
    df_cutoff["min"] = grouper_obj.min()
    df_cutoff["max"] = grouper_obj.max()
    df_cutoff["median"] = grouper_obj.median()
    df_cutoff.reset_index(inplace=True)
    df_cutoff = df_cutoff[~df_cutoff["median"].isna()]

    # add missing to the cutoff table
    if include_missing:
        df_cutoff.at[missing_idx, "min"] = np.nan
        df_cutoff.at[missing_idx, "max"] = np.nan
        df_cutoff.at[missing_idx, "median"] = np.nan

    # generate predictions on median of each bin
    grid = df_cutoff["median"].to_list()
    preds = {}

    for x in grid:
        df_[feature] = x
        y = np.average(model.predict(df_[features]))
        if np.isnan(x):
            df_cutoff.loc[df_cutoff["median"].isnull(), "pred"] = y
        else:
            df_cutoff.loc[df_cutoff["median"] == x, "pred"] = y

    return df_cutoff


def pdp_w_impact_table(
    model: lgb.Booster,
    df: pd.DataFrame,
    features: list,
    num_cuts=20,
    include_missing=True,
):
    """
    providing a fixed feature, compute partial dependency by value quantiles

    impact = pred - mean_prediction
    if target = bad, then we want features with highest "impact"
    - in this case it might be more appropriate to refer to it as negative impact

    @params model: lgb.Booster or model object with a predict function
        predict generates a 1d Series containing probabilities
        one for each record
    @params df: pd.DataFrame - dataset for pdp computation
    @params features: list[str] - feature list for model prediction
    @params num_cuts: int - number of value cutoffs/quantiles
    @params include_missing: bool - whether to include a row indicate missing values

    @return pdp_dict: dictionary containing pdp tables with impact
    """
    from collections import OrderedDict
    from tqdm import tqdm

    try:
        getattr(model, "predict")
    except AttributeError as e:
        raise AttributeError(e)

    pdp_dict = OrderedDict()
    pred_mean = np.mean(model.predict(df[features]))

    for f in tqdm(features):
        tbl_ = partial_dependency_(
            model, df, f, features, num_cuts=num_cuts, include_missing=include_missing
        )
        # impact = pred - mean prediction
        tbl_["impact"] = tbl_["pred"] - pred_mean
        pdp_dict[f] = tbl_

    return pdp_dict


def plot_pdp_w_impact(
    pdp_dict,
    features,
    ncols=3,
    figsize=None,
    linewidth=3,
    fig=None,
    axs=None,
    max_n_ticks=15,
    **kwargs
):
    """
    Plot partial dependency table with impact
    """
    import math

    nrows = math.ceil(len(features) / ncols)

    if figsize is None:
        figsize = (ncols * 8, nrows * 8)

    if fig is None and axs is None:
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for f, ax in zip(features, axs.flatten()):
        tbl = pdp_dict[f].copy()
        missing = tbl[tbl["median"].isna()]
        tbl = tbl[~tbl["median"].isna()]
        rows = range(0, len(tbl), max(len(tbl) // max_n_ticks, 1))
        rows = sorted(list(set(list(rows) + [len(tbl) - 1])))
        tbl = tbl.iloc[rows]

        # plot pdp
        sns.lineplot(tbl["median"].astype(str), tbl["impact"], ax=ax)

        # plot missing
        sns.scatterplot(missing["median"].astype(str), missing["impact"], ax=ax)

        # plot mean prediction line
        ax.axhline(y=0, linestyle="--")
        ax.tick_params(axis="x", labelrotation=45)

        ax.set_ylabel("impact")
        ax.set_xlabel("values")
        ax.title.set_text(f)

    return fig, axs


def build_aa_code_pdp(
    df: pd.DataFrame,
    pdp_dict: dict,
    aa_df: pd.DataFrame,
    inq_fts: list = [],
    aa_code_valid_col="AA_code_valid",
    aa_code_special_col="AA_code_special_value",
    special_list: list = [],
    missing_idx: int = None,
):
    """
    df is a single record

    inq_fts: features about number of inquiries used

    @params df: pd.DataFrame - input data in a df format
        for example, in a batched dataset "data", an input df would be
        the result of the following
            input_row/df = data.iloc[0].to_frame().T
    @params pdp_dict: Dict[str, pd.DataFrame]
        product of our previous pdp dictionary object

    @params aa_df: pd.DataFrame - df that contains AA code explanations
        must have "attr", "AA_code_valid", "AA_code_missing" columns
        - attr: model attribute names
        - AA_code_valid
        - AA_code_missing

    """

    features = pdp_dict.keys()
    impact_df = pd.DataFrame()
    grid = {}

    for i, f in enumerate(features):
        tbl = pdp_dict[f]
        if missing_idx:
            # this line is surprisingly expensive. only filter if needed
            raise NotImplementedError("Having tested cases with missing value")
            tbl_no_missing = tbl[tbl.index != missing_idx]
        else:
            tbl_no_missing = tbl

        if df[f].isna().sum() >= 1:
            pos = missing_idx
        else:
            a = np.array(df[f].values) - np.array(tbl_no_missing["min"].to_list())
            b = np.ma.MaskedArray(a, a < 0, fill_value=99999999999)
            pos = np.ma.argmin(b)

        impact_df.loc[i, "attr"] = f
        impact_df.loc[i, "pos"] = pos
        impact_df.loc[i, "value"] = df[f].item()
        impact_df.loc[i, "impact"] = tbl.loc[pos]["impact"]

    impact_df.sort_values(by="impact", ascending=False, inplace=True)
    impact_df.loc[impact_df["attr"].isin(inq_fts), "is_inquiry"] = True
    impact_df["inquiry_impact"] = 0

    # inquiry is counted as important if the impact is greater than 0
    impact_df.loc[
        impact_df["is_inquiry"] & (impact_df["impact"] > 0), "inquiry_impact"
    ] = 1

    pos_impact = impact_df[impact_df.impact > 0]
    pos_impact_aa = pos_impact.merge(aa_df, how="left", on="attr")

    pos_impact_aa.sort_values(
        by=["inquiry_impact", "impact"], ascending=False, inplace=True
    )

    n = 4
    if impact_df["inquiry_impact"].max() >= 1:
        n = 5

    pos_impact_aa["aa"] = np.where(
        pos_impact_aa.attr.isin(special_list),
        pos_impact_aa[aa_code_special_col],
        pos_impact_aa[aa_code_valid_col],
    )

    aa_top = pos_impact_aa.drop_duplicates(["aa"], keep="first")[:n]
    aa_top.sort_values(by=["impact"], ascending=False, inplace=True)

    return impact_df, aa_top
