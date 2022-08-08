"""
Functions for binning process
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2018.

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from .ctree import CTree
from .util import _binary
from .util import _check_input
from .util import _check_input_table
from .util import _rounding
from .util import _separated_data


_SPECIAL_POLICY_OPTIONS = ["join", "separate", "binning"]
_WOE_POLICY_OPTIONS = ["empirical", "worst", "zero"]

_GRP_NAME = "GRP_{}"
_WOE_NAME = "WOE_{}"
_MTR_NAME = "MTR_{}"

_PLOT_OUTPUTS = ["save", "term", "terminal"]
_PLOT_TYPES = ["pd", "woe", "default"]


def apply(df, value_name, target_name, splits, grp_values, special_values=[],
          splits_specials=[], woe=False, check_input=False,
          return_group=False):
    """
    Apply transformation given a dataframe and a list of splits.

    This method should not be used when using OptimalGrouping, use
    ``grmlab.data_processing.feature_binning.OptimalGrouping.transform()``
    instead.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to perform optimal grouping.

    value_name : str
        The variable name.

    target_name : str
        The name of the target variable.

    splits : list
        List of split points.

    grp_values : list
        List of values for each bucket or group.

    special_values : list (default=[])
        List of special values to be considered.

    splits_special : list (default=[])
        List of split points for special values data.

    woe : boolean (default=False)
        Add Weight-of-Evidence column.

    check_input : boolean (default=False)
        Option to perform several input checking.

    return_group : boolean (default=False)
        Return column with group id's.

    Returns
    -------
    transform : pandas.DataFrame
    """

    # general checking
    if check_input:
        _check_input(df, value_name, target_name)

    # check grp_values
    if len(grp_values) < len(splits) + 1:
        raise ValueError("Lenght of grp_values < length splits + 1.")

    # convert data frame into two numpy arrays
    x = df[value_name].values

    # indexes with special values in value_name
    idx_spec = np.isin(x, special_values)

    # header names
    grp_name = _GRP_NAME.format(value_name)
    if woe:
        metric_name = _WOE_NAME.format(value_name)
    else:
        metric_name = _MTR_NAME.format(value_name)

    n_splits = len(splits)

    if not n_splits:
        # no splits
        df[grp_name] = 0

        # indexes with NaN in value_name
        idx_nan = pd.isnull(x)

    elif isinstance(splits[0], np.ndarray):

        # categorical / nominal variables return groups as numpy.ndarray
        for idx, split in enumerate(splits):
            split = np.asarray(split)  # force casting
            df.loc[df[value_name].isin(split), grp_name] = idx

        # indexes with NaN in value_name
        idx_nan = pd.isnull(x)
    else:
        # make sure splits are sorted and these are unique
        splits = np.unique(np.asarray(splits))
        splits = splits[::-1]

        # dataframe comparison operations
        df[grp_name] = -1
        df.loc[(df[value_name] > splits[-1]), grp_name] = n_splits

        for idx, split in enumerate(splits):
            df.loc[(df[value_name] <= split), grp_name] = n_splits - (idx+1)

        # indexes with NaN in value_name
        idx_nan = np.isnan(x)

        # account for > group
        n_splits += 1

    # special values
    if splits_specials:
        for idx, split in enumerate(splits_specials):
            if isinstance(split, np.ndarray):
                df.loc[df[value_name].isin(split), grp_name] = n_splits
            else:
                df.loc[df[value_name] == split, grp_name] = n_splits
            n_splits += 1
    else:
        df.loc[idx_spec, grp_name] = n_splits
        n_splits += 1

    # missing values
    df.loc[idx_nan, grp_name] = n_splits

    if return_group:
        return df[grp_name]
    else:
        metric_map = dict(enumerate(grp_values[:-1]))  # exclude total row
        df[metric_name] = df[grp_name].map(metric_map)

    return df


def _iv_table(x, y, weight, xnan, ynan, wnan, xspec, yspec, wspec, splits, special_policy,
              missing_woe_policy, special_woe_policy, return_iv=False):
    """Generate binning table dataframe for binary target."""

    # options checks
    if special_policy not in _SPECIAL_POLICY_OPTIONS:
        raise ValueError("special_policy option not supported.")

    if missing_woe_policy not in _WOE_POLICY_OPTIONS:
        raise ValueError("missing_woe_policy option not supported.")

    if special_woe_policy not in _WOE_POLICY_OPTIONS:
        raise ValueError("special_woe_policy option not supported.")

    # categorical / nominal flag
    is_cat = False

    # splits for special values
    splits_specials = []

    # indicator arrays
    y0 = (y == 0)
    y1 = ~y0

    n = sum(weight)
    total_nonevent = sum(weight[y0])
    total_event = n - total_nonevent

    nonevent = []
    event = []
    records = []

    n_splits = len(splits)

    if not n_splits:
        # no splits
        nonevent.append(sum(weight[y0]))
        event.append(sum(weight[y1]))
        records.append(nonevent[-1] + event[-1])

    elif isinstance(splits[0], (list, np.ndarray)):
        # categorical / nominal variables return groups as numpy.ndarray
        is_cat = True

        # all splits => groups
        for split in splits:
            # idx = np.isin(x, split)
            idx = pd.Series(x).isin(split).values
            nonevent.append(sum(weight[idx & y0]))
            event.append(sum(weight[idx & y1]))
            records.append(nonevent[-1] + event[-1])
    else:
        # numerical variable
        n_splits += 1
        splits = np.unique(np.asarray(splits))
        bands = np.insert(splits, 0, np.min(x))
        bands = np.append(bands, np.max(x))

        # first split
        idx = (x <= splits[0])
        nonevent.append(sum(weight[idx & y0]))
        event.append(sum(weight[idx & y1]))
        records.append(nonevent[-1] + event[-1])

        # subsequent splits
        for i in range(1, len(bands) - 1):
            idx = (x > bands[i]) & (x <= bands[i+1])
            nonevent.append(sum(weight[idx & y0]))
            event.append(sum(weight[idx & y1]))
            records.append(nonevent[-1] + event[-1])

    # special values
    n_spec = sum(wspec)
    y0 = (yspec == 0)
    y1 = (yspec == 1)

    n_specials = np.unique(xspec)

    if n_spec and special_policy == "separate":
        for ds in n_specials:
            idx = np.isin(xspec, ds)
            nonevent.append(sum(wspec[idx & y0]))
            event.append(sum(wspec[idx & y1]))
            records.append(nonevent[-1] + event[-1])

        splits_specials = list(n_specials)

    elif n_spec and special_policy == "binning" and len(n_specials) > 1:

        ct = CTree(dtype="nominal")
        ct.fit(xspec, yspec)
        splits_specials = ct.splits

        for split in splits_specials:
            idx = np.isin(xspec, split)
            nonevent.append(sum(wspec[idx & y0]))
            event.append(sum(wspec[idx & y1]))
            records.append(nonevent[-1] + event[-1])

        splits_specials = splits_specials

    else:
        # special_values="join"
        nonevent.append(sum(wspec[y0]))
        event.append(sum(wspec[y1]))
        records.append(n_spec)

    # missing values
    n_nan = sum(wnan)
    y0 = (ynan == 0)
    y1 = (ynan == 1)
    nonevent.append(sum(wnan[y0]))
    event.append(sum(wnan[y1]))
    records.append(n_nan)

    # total counters
    total_nonevent = sum(nonevent)
    total_event = sum(event)
    total_records = n + n_nan + n_spec

    # number of special groups generated
    n_grp_specials = max(0, len(records) - 1 - n_splits)

    # compute default rate (probability of default - PD)
    default_rate = [event[i] / records[i] if records[i] else "-"
                    for i in range(len(records))]

    if not return_iv:
        # prepare special rows
        if n_spec and special_policy == "separate":
            str_special = ["{}".format(sp) for sp in n_specials]
        elif n_spec and special_policy == "binning" and len(n_specials) > 1:
            str_special = ["{}".format(", ".join((map(str, sorted(sp)))))
                           for sp in splits_specials]
        else:
            str_special = ["Special"]

        # prepare rest of rows and data
        if len(splits) == 0:
            split_str = ["No splits"] + str_special + ["Missing"]
        elif is_cat:
            split_str = ["{}".format(", ".join((map(str, sorted(split)))))
                         for split in splits]
            split_str += str_special + ["Missing"]
        else:
            # clever rounding
            rd = _rounding(splits)

            split_str = ["<= {}".format(round(split, rd)) for split in splits]
            split_str.append("> {}".format(round(splits[-1], rd)))
            split_str += str_special + ["Missing"]

        # create dataframe
        columns = ["Splits", "Records count", "Non-event count", "Event count"]
        ivtdata = [split_str, records, nonevent, event]
        ivtable = pd.DataFrame(dict(zip(columns, ivtdata)), columns=columns)

        # add default rate
        ivtable["Default rate"] = default_rate

    # add woe and iv
    nonevent_p = np.array(nonevent) / total_nonevent
    event_p = np.array(event) / total_event

    woe = []
    iv = []
    for i in range(len(records)):
        if nonevent_p[i] * event_p[i]:
            _w = np.log(nonevent_p[i] / event_p[i])
            _iv = _w * (nonevent_p[i] - event_p[i])
            woe.append(_w)
            iv.append(_iv)
        else:
            woe.append(0)
            iv.append(0)

    # apply special woe policy
    if special_woe_policy == "zero":
        woe[n_splits:-1] = [0] * n_grp_specials
        iv[n_splits:-1] = [0] * n_grp_specials
    elif special_woe_policy == "worst":
        # iv and woe corresponding to the group with the highest default rate
        if n_splits:
            idx = np.argmax(default_rate[:n_splits])
            woe[n_splits:-1] = [woe[idx]] * n_grp_specials
            iv[n_splits:-1] = [iv[idx]] * n_grp_specials
        else:
            # no splits, only one group
            woe[n_splits:-1] = [woe[0]] * n_grp_specials
            iv[n_splits:-1] = [iv[0]] * n_grp_specials

    # apply missing woe policy
    if missing_woe_policy == "zero":
        woe[-1] = 0
        iv[-1] = 0
    elif missing_woe_policy == "worst":
        # iv and woe corresponding to the group with the highest default rate
        if n_splits:
            idx = np.argmax(default_rate[:n_splits])
            woe[-1] = woe[idx]
            iv[-1] = iv[idx]
        else:
            # no splits, only one group
            woe[-1] = woe[0]
            iv[-1] = iv[0]

    # compute final IV after applying policies
    ivsum = sum([_iv if _iv != "-" else 0 for _iv in iv])

    if return_iv:
        return records, ivsum, woe, splits_specials
    else:
        ivtable["WoE"] = woe
        ivtable["IV"] = iv
        iv = ivsum

    # add total row
    ivtable.loc["Total"] = ["", total_records, total_nonevent, total_event,
                            total_event / total_records, "", iv]

    return ivtable, splits_specials


def _general_table(x, y, xnan, ynan, xspec, yspec, splits, special_policy):
    """Generate binning table dataframe for continuous target."""
    if special_policy not in _SPECIAL_POLICY_OPTIONS:
        raise ValueError("special_policy option not supported.")

    # categorical / nominal flag
    is_cat = False

    # splits for special values
    splits_specials = []

    sum_grp = []
    mean_grp = []
    median_grp = []
    std_grp = []
    max_grp = []
    min_grp = []
    records = []
    zeros_grp = []

    if len(splits) == 0:
        # no splits
        records.append(len(y))
        sum_grp.append(np.sum(y))
        mean_grp.append(np.mean(y))
        median_grp.append(np.median(y))
        std_grp.append(np.std(y))
        max_grp.append(np.max(y))
        min_grp.append(np.min(y))
        zeros_grp.append(np.count_nonzero(y == 0))

    elif isinstance(splits[0], (list, np.ndarray)):
        # categorical / nominal variables return groups as numpy.ndarray
        is_cat = True

        # all splits => groups
        for split in splits:
            # ydx = y[pd.Series(x).isin(split).values]
            ydx = y[pd.Series(x).isin(split).values]
            sum_grp.append(np.sum(ydx))
            mean_grp.append(np.mean(ydx))
            median_grp.append(np.median(ydx))
            std_grp.append(np.std(ydx))
            max_grp.append(np.max(ydx) if len(ydx) else np.nan)
            min_grp.append(np.min(ydx) if len(ydx) else np.nan)
            records.append(len(ydx))
            zeros_grp.append(np.count_nonzero(ydx == 0))
    else:
        # numerical variables
        splits = np.unique(np.asarray(splits))
        bands = np.insert(splits, 0, np.min(x))
        bands = np.append(bands, np.max(x))

        # first split
        ydx = y[(x <= splits[0])]
        sum_grp.append(np.sum(ydx))
        mean_grp.append(np.mean(ydx))
        median_grp.append(np.median(ydx))
        std_grp.append(np.std(ydx))
        max_grp.append(np.max(ydx) if len(ydx) else np.nan)
        min_grp.append(np.min(ydx) if len(ydx) else np.nan)
        records.append(len(ydx))
        zeros_grp.append(np.count_nonzero(ydx == 0))

        # subsequent splits
        for i in range(1, len(bands) - 1):
            ydx = y[(x > bands[i]) & (x <= bands[i+1])]
            sum_grp.append(np.sum(ydx))
            mean_grp.append(np.mean(ydx))
            median_grp.append(np.median(ydx))
            std_grp.append(np.std(ydx))
            max_grp.append(np.max(ydx) if len(ydx) else np.nan)
            min_grp.append(np.min(ydx) if len(ydx) else np.nan)
            records.append(len(ydx))
            zeros_grp.append(np.count_nonzero(ydx == 0))

    # special values
    n_spec = len(yspec)
    n_specials = np.unique(xspec)

    if n_spec and special_policy == "separate":
        for ds in n_specials:
            ydx = y[np.isin(x, ds)]
            sum_grp.append(np.sum(ydx))
            mean_grp.append(np.mean(ydx))
            median_grp.append(np.median(ydx))
            std_grp.append(np.std(ydx))
            max_grp.append(np.max(ydx) if len(ydx) else np.nan)
            min_grp.append(np.min(ydx) if len(ydx) else np.nan)
            records.append(len(ydx))
            zeros_grp.append(np.count_nonzero(ydx == 0))

            splits_specials = list(n_specials)

    elif n_spec and special_policy == "binning" and len(n_specials) > 1:
        ct = CTree(dtype="nominal")
        ct.fit(xspec, yspec)
        splits_specials = ct.splits

        for split in splits_specials:
            # ydx = y[np.isin(x, split)]
            ydx = y[pd.Series(x).isin(split).values]
            sum_grp.append(np.sum(ydx))
            mean_grp.append(np.mean(ydx))
            median_grp.append(np.median(ydx))
            std_grp.append(np.std(ydx))
            max_grp.append(np.max(ydx) if len(ydx) else np.nan)
            min_grp.append(np.min(ydx) if len(ydx) else np.nan)
            records.append(len(ydx))
            zeros_grp.append(np.count_nonzero(ydx == 0))

        splits_specials = splits_specials

    else:
        # special_values="join"
        sum_grp.append(np.sum(yspec) if n_spec else np.nan)
        mean_grp.append(np.mean(yspec) if n_spec else np.nan)
        median_grp.append(np.median(yspec) if n_spec else np.nan)
        std_grp.append(np.std(yspec) if n_spec else np.nan)
        max_grp.append(np.max(yspec) if n_spec else np.nan)
        min_grp.append(np.min(yspec) if n_spec else np.nan)
        records.append(n_spec)
        zeros_grp.append(np.count_nonzero(yspec == 0))

    # missing values
    n_nan = len(ynan)
    sum_grp.append(np.sum(ynan) if n_nan else None)
    mean_grp.append(np.mean(ynan) if n_nan else None)
    median_grp.append(np.median(ynan) if n_nan else np.nan)
    std_grp.append(np.std(ynan) if n_nan else np.nan)
    max_grp.append(np.max(ynan) if n_nan else np.nan)
    min_grp.append(np.min(ynan) if n_nan else np.nan)
    records.append(n_nan)
    zeros_grp.append(np.count_nonzero(ynan == 0))

    total_records = sum(records)

    # prepare special rows
    if n_spec and special_policy == "separate":
        str_special = ["{}".format(sp) for sp in n_specials]
    elif n_spec and special_policy == "binning" and len(n_specials) > 1:
        str_special = ["{}".format(", ".join((map(str, sorted(sp)))))
                       for sp in splits_specials]
    else:
        str_special = ["Special"]

    # prepare rest of rows and data
    if len(splits) == 0:
        split_str = ["No splits"] + str_special + ["Missing"]
    elif is_cat:
        split_str = ["{}".format(", ".join((map(str, sorted(split)))))
                     for split in splits]
        split_str += str_special + ["Missing"]
    else:
        # clever rounding
        rd = _rounding(splits)

        split_str = ["<= {}".format(round(split, rd)) for split in splits]
        split_str.append("> {}".format(round(splits[-1], rd)))
        split_str += str_special + ["Missing"]

    # create dataframe
    records_perc = np.array(records) / total_records
    columns = ["Splits", "Records count", "Records (%)", "Sum", "Mean",
               "Median", "Std", "Min", "Max", "Zeros"]
    tdata = [split_str, records, records_perc, sum_grp, mean_grp, median_grp,
             std_grp, min_grp, max_grp, zeros_grp]
    gtable = pd.DataFrame(dict(zip(columns, tdata)), columns=columns)

    total_sum_grp = np.sum(y)
    total_mean_grp = np.mean(y)
    total_median_grp = np.median(y)
    total_std_grp = np.std(y)
    total_max_grp = np.max(y)
    total_min_grp = np.min(y)
    total_zeros_grp = np.count_nonzero(y == 0)
    gtable.loc["Total"] = ["", total_records, 1, total_sum_grp, total_mean_grp,
                           total_median_grp, total_std_grp, total_min_grp,
                           total_max_grp, total_zeros_grp]

    return gtable, splits_specials


def table(values, target, splits, sample_weight=None,
          special_handler_policy="binning",
          special_values=[], special_woe_policy="empirical",
          missing_woe_policy="empirical", check_input=False):
    """General function to generate binning table given data and splits.

    Parameters
    ----------
    values : list or numpy.ndarray
        The variable data samples.

    target : list or numpy.ndarray
        The target samples.

    splits : list or numpy.ndarray
        List of split points.

    sample_weight : numpy.ndarray (default=None)
        Individual weights for each sample.

    special_handler_policy : str (default="join")
        Method to handle special values. Options are "join", "separate" and
        "binning". Option "join" creates an extra bucket containing all special
        values. Option "separate" creates an extra bucket for each special
        value. Option "binning" performs feature binning of special values
        using ``grmlab.data_processing.feature_binning.CTree`` in order to
        split special values if these are significantly different.

    special_woe_policy : str (default=empirical)
        Weight-of-Evidence (WoE) value to be assigned to special values
        buckets. Options supported are: "empirical", "worst", "zero". Option
        "empirical" assign the actual WoE value. Option "worst" assigns the WoE
        value corresponding to the bucket with the highest event rate. Finally,
        option "zero" assigns value 0.

    missing_woe_policy : str (default="empirical")
        Weight-of-Evidence (WoE) value to be assigned to missing values bucket.
        Options supported are: "empirical", "worst", "zero". Option "empirical"
        assign the actual WoE value. Option "worst" assigns the WoE value
        corresponding to the bucket with the highest event rate. Finally,
        option "zero" assigns value 0.

    check_input : boolean (default=True)
        Option to perform several input checking.

    Returns
    -------
    binning_table : pandas.DataFrame
    """

    # convert dataframe into two numpy arrays
    x = values
    y = target
    weight = sample_weight

    x, y, weight, xnan, ynan, wnan, xspec, yspec, wspec = _separated_data(x, y, weight, special_values)

    # if binary compute iv table otherwise general binning table
    if _binary(y):
        return _iv_table(x, y, weight, xnan, ynan, wnan, xspec, yspec, wspec,
                         splits, special_handler_policy, missing_woe_policy,
                         special_woe_policy)
    else:
        return _general_table(x, y, xnan, ynan, xspec, yspec, splits,
                              special_handler_policy)


def plot(df, splits, plot_type="pd", plot_bar_type="event", percentage=False,
         others_group=False, output="term", outfile=None, check_input=False,
         lb_feature=None):
    """Plot binning table result.

    This function requires a table generated by using
    ``grmlab.data_processing.feature_binning.table()``.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to perform optimal grouping.

    plot_type : str (default="pd")
        The measure to show in y-axis. Options are: "pd" and "woe".

    plot_bar_type : str (default="event")
        The count value to show in barplot. Options are: "all", "event"
        and "nonevent".

    percentage : boolean (default=False)
        Whether to show y-axis with percentages.

    others_group : boolean (default=False)
        A flag to indicate whether an extra group with values not sufficiently
        representative. This option is available for dtypes
        "categorical" and "nominal".

    output : str (default="term")
        Plot output options. Supported options are "save" and "term".

    outfile : str or None (default=None)
        Outfile path necessary if ``output=="save"``.

    check_input : boolean (default=False)
        Option to perform several input checking.

    lb_feature: str or None (default=None)
        The name of the feature to be plotted.

    See also
    --------
    table
    """
    _df = df.copy()

    # check dataframe comes from table() function.
    if check_input:
        _check_input_table(_df)

    if plot_type not in _PLOT_TYPES:
        raise ValueError("plot_type option not supported.")

    if plot_bar_type not in ("event", "nonevent", "all"):
        raise ValueError("plot_bar_type option no supported.")

    if output not in _PLOT_OUTPUTS:
        raise ValueError("output option not supported.")

    if output == "save" and not outfile:
        raise ValueError("output type save, provide an outfile.")

    # number of splits and type
    n_splits = len(splits)

    if n_splits and not isinstance(splits[0], np.ndarray):
        n_splits += 1

    if not n_splits:
        n_splits = 1

    # set plot colors and options
    _BLACK_COLOR = "#47476b"
    _BLUE_COLOR = "#66CDAA"
    _GREEN_COLOR = "#2E8B57"
    _GREY_COLOR = "#9494b8"
    _PURPLE_COLOR = "#cc66ff"
    _LINE_STYLE = "o-"

    _FONTSIZE = 16
    _LEGEND_LOCATION = 'upper center'

    fig, ax = plt.subplots(1, 1)

    # retrieve data from df
    if plot_type in ["woe", "pd"]:
        if plot_bar_type is "event":
            lb_records = "Event count"
        elif plot_bar_type is "nonevent":
            lb_records = "Non-event count"
        elif plot_bar_type is "all":
            lb_records = "Records count"

        lb_feature = "WoE" if plot_type == "woe" else "Default rate"
        feature = _df[lb_feature].values[:-1]
    else:
        lb_records = "Records count"
        if lb_feature is None:
            lb_feature = "Mean"
        else:
            if lb_feature not in ["Mean", "Median"]:
                raise ValueError("feature not supported.")

    feature = _df[lb_feature].values[:-1]
    feature[feature == "-"] = np.nan # check this
    records = _df[lb_records].values[:-1]

    # splits, specials and nan
    feature_grp = feature[:n_splits]
    special_grp = feature[n_splits:-1]
    missing_grp = feature[-1]

    # plot records and overwrite special and missing bars
    rng = range(len(records))
    barlst = ax.bar(rng, records, color=_BLUE_COLOR, label=lb_records)
    for i in range(n_splits, len(records)-1):
        barlst[i].set_color(_GREY_COLOR)
    barlst[-1].set_color(_BLACK_COLOR)

    if others_group:
        barlst[n_splits-1].set_color(_PURPLE_COLOR)

    ax2 = ax.twinx()

    # plot feature
    if percentage or plot_type == "pd":
        feature_grp *= 100
        special_grp *= 100
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    # plot split groups
    if others_group:
        # plot other group WoE
        ax2.plot(n_splits-1, feature_grp[n_splits-1], "o", color="purple")
        feature_grp = feature_grp[:-1]

    ax2.plot(range(len(feature_grp)), feature_grp, _LINE_STYLE,
             color=_GREEN_COLOR, label=lb_feature)

    # special groups
    ax2.plot(range(n_splits, n_splits+len(special_grp)), special_grp, "o",
             color="black")

    # missing group
    ax2.plot(len(records)-1, missing_grp, "o", color="black")

    ax2.set_ylabel(lb_feature, fontsize=_FONTSIZE)
    ax2.set_xlabel("GRP", fontsize=_FONTSIZE)

    # prepare legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines += lines2
    labels += labels2

    ax2.legend(lines, labels, loc=_LEGEND_LOCATION, bbox_to_anchor=(0.5, -0.1),
               fancybox=True, shadow=True, ncol=2)

    # custom x axis tickers
    ax2.xaxis.set_major_locator(mtick.MultipleLocator(1))
    plt.xlim(-0.75, len(records)-0.5)

    # output
    if output in ["term", "terminal"]:
        plt.show()
    else:
        fig.savefig(outfile)
