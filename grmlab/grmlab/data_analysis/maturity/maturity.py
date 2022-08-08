"""
Maturity analysis
"""

# Authors: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2020.

from scipy.stats import norm
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .util import infer_change_points
from ...core.base import GRMlabBase
from ...core.exceptions import NotRunException


class MaturityAnalysis(GRMlabBase):
    """
    Analyzes the maturity of the good status.

    Parameters
    ----------
    method: str (default="piecewise")
        Method used for the maturity analysis. The options are: accumulative,
        or piecewise.

    target : str or None (default=None)
        The name of the variable flagged as target.

    date : str or None (default=None)
        The name of the variable flagged as date.

    default_date : str or None (default=None)
        The name of the variable flagged as minimum date of default.

    date_format : str or None (default="%Y%m")
        The strftime to parse time, eg “%d/%m/%Y”, note that “%f” will parse
        all the way up to nanoseconds. See strftime documentation for more
        information on choices:
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.

    beta: float (default=None)
        Cost for each additional segment. For piecewise method only.

    gamma: float (default=None)
        Penalization for longer segments. For piecewise method only.

    min_points: int (default=3)
        Minimum number of data points in one segment. For piecewise method
        only.

    max_p_value: float (default=0.05)
        Max p-value for the test comparing the slopes of consecutive segments.
        For piecewise method only.

    verbose: int or Boolean (default=False)
        Controls verbosity of output.
    """
    def __init__(self, method="piecewise", target=None, date=None,
                 default_date=None, date_format=None, beta=None, gamma=0,
                 min_points=3, max_p_value=0.05, verbose=False):
        # initial values database
        self.target = target
        self.date = date
        self.default_date = default_date
        self.date_format = date_format
        # initial values maturity
        self.method = method
        self.beta = beta
        self.gamma = gamma
        self.min_points = min_points
        self.max_p_value = max_p_value
        self.verbose = verbose

        # run
        self.y = None
        self.x = None
        self.date_vec = None
        # piecewise
        self.buckets_y = None
        self.buckets_x = None
        self.split_points = None
        self.clean_buckets_y = None
        self.clean_buckets_x = None
        # accumulative
        self.df_accum = None
        self._n_samples = None
        self.accum_06 = None
        self.accum_08 = None
        # result
        self.maturity = None

        # flags
        self._is_run = False
        self._is_transformed = False

        if self.date_format is None:
            self.date_format = "%Y%m"

    def run(self, df=None, y=None, x=None):
        """Run the maturity analysis.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw dataset. If df is not None, inputs y and x will not be
            considered.

        y: list or numpy.array
            y values of the data.

        x: list or numpy.array (default=None)
            x values of the data. If None, equal unit spaces will be
            calculated.
        """

        if (df is None) and (y is None):
            raise TypeError("Neither df, nor y is informed."
                            "At least one of them must be informed.")
        elif (df is None) and (self.method == "accumulative"):
            raise TypeError("df must be informed for accumulative method.")
        elif ((df is not None) and
              ((self.date is None) or (self.target is None))):
            raise TypeError("date and target must be informed if df is given")
        elif ((self.method == "accumulative") and (self.default_date is None)):
            raise TypeError("default_date must be informed for "
                            "accumulative method")

        # check method
        if self.method not in ["accumulative", "piecewise"]:
            raise ValueError("Method {} no supported".format(
                             self.method))

        if self.method == "piecewise":
            # check min_points
            if not isinstance(self.min_points, int):
                raise TypeError("min_points must be integer type")
            elif self.min_points < 2:
                raise ValueError("min_points must be larger than 1")

            # check max_p_value
            if isinstance(self.max_p_value, float):
                if not ((self.max_p_value > 0.0) and (self.max_p_value < 1.0)):
                    raise ValueError("max_p_value be in range (0,1).")
            else:
                raise TypeError("max_p_value must be float.")

            # check beta
            if self.beta is not None:
                if not isinstance(self.beta, (float, int)):
                    raise TypeError("beta must be float or int.")

            # check gamma
            if self.gamma is not None:
                if not isinstance(self.gamma, (float, int)):
                    raise TypeError("gamma must be float or int.")

        self._run(df, y, x)

        self._is_run = True

        return self.maturity

    def plot_result(self):
        """Plots the results for any of the methods."""
        if not self._is_run:
            raise NotRunException(self, "run")

        if self.method == "piecewise":
            self._plot_piecewise()
        elif self.method == "accumulative":
            self._plot_accumulative()

    def _accumulative_method(self):
        """Calculates the maturity with the accumulative method."""
        max_months = np.max(self.y)  # max maturity
        n_bads = self.y.shape[0]
        count = 0
        cum_vec = []
        cum_perc = []
        bads_vec = []
        bads_perc_vec = []
        dr_vec = []
        # loop through months to maturity (MTM)
        for i in range(max_months + 1):
            bads = self.y[self.y == i].shape[0]
            count += bads
            bads_vec.append(bads)  # number of bads on that MTM
            bads_perc_vec.append(bads/n_bads)  # perd of bads on that MTM
            cum_vec.append(count)  # accumulative number of bads
            cum_perc.append(count/n_bads)  # accumulative % of bads
            dr_vec.append(count/self._n_samples)
        # df with all the calculated vectors
        self.df_accum = pd.DataFrame(data={"bads_n": bads_vec,
                                           "bads_perc": bads_perc_vec,
                                           "accumulated_n": cum_vec,
                                           "accumulated_per": cum_perc,
                                           "default_rate": dr_vec})
        # value of MTM at 60% accumulated bads
        self.accum_06 = np.argmin(np.abs(np.array(cum_perc) - 0.6))
        # value of MTM at 80% accumulated bads
        self.accum_08 = np.argmin(np.abs(np.array(cum_perc) - 0.8))

        if self.verbose:
            print("0.6% accumulated defaults at {} months".format(
                self.accum_06))
            print("0.8% accumulated defaults at {} months".format(
                self.accum_08))

        self.maturity = self.accum_06
        if self.verbose:
            self._plot_accumulative()

    def _bucket_statistics(self, y_buckets, x_buckets):
        """Calculates the linear regression and statistics for each segment.
        """
        mult_reg = []
        slopes_vec = []
        var_vec = []
        n_vec = []
        # loop through all the segments
        for i, bkt in enumerate(y_buckets):
            y_reg = bkt  # y-values in the i-th segment
            x_reg = x_buckets[i].reshape(-1, 1)  # x-values in the i-th segment

            # Fit the linear regression for the segment
            reg = LinearRegression(fit_intercept=True).fit(x_reg, y_reg)
            mult_reg += list(reg.predict(x_reg))

            # Statistics
            # Sum of Squared Errors
            SSE = sum(np.power(y_reg - reg.predict(x_reg), 2))
            x_error = sum(np.power(x_reg - np.mean(x_reg), 2))
            # Error variance of the slope
            var_vec.append((SSE)/((len(y_reg)-2)*(x_error)))
            # Value of the slope
            slopes_vec.append(reg.coef_[0])
            # number of elements in the segment
            n_vec.append(len(y_reg))

        return mult_reg, slopes_vec, var_vec, n_vec

    def _compliant_buckets(self, slopes_vec, var_vec):
        """Check if consecutives segments are statistically different"""

        Z_critical = norm.ppf(1-self.max_p_value)  # test critical value

        self.clean_buckets_y = [self.buckets_y[-1]]
        self.clean_buckets_x = [self.buckets_x[-1]]
        j = 0  # index of self.clean_buckets_y
        i = len(slopes_vec)-1  # start index. By the most recent segment.
        first_segment = True
        while i > 0:
            if (var_vec[i] + var_vec[i-1]) == 0.0:
                if (slopes_vec[i-1]-slopes_vec[i]) == 0.0:
                    Z = 0
                else:
                    Z = 9e999
            else:
                Z = np.abs((slopes_vec[i-1]-slopes_vec[i]) /
                           np.sqrt(var_vec[i] + var_vec[i-1]))  # Statist
            if Z >= Z_critical:
                # decreasing slope is only check for the first segment
                if first_segment:
                    if (slopes_vec[i] < 0):
                        if (slopes_vec[i] < slopes_vec[i-1]):
                            j += 1  # new clean bucket
                            self.clean_buckets_y.append(self.buckets_y[i-1])
                            self.clean_buckets_x.append(self.buckets_x[i-1])
                            first_segment = False
                        else:
                            self.clean_buckets_y[j] = np.append(
                                self.buckets_y[i-1], self.clean_buckets_y[j])
                            self.clean_buckets_x[j] = np.append(
                                self.buckets_x[i-1], self.clean_buckets_x[j])
                    else:
                        self.clean_buckets_y[j] = np.append(
                            self.buckets_y[i-1], self.clean_buckets_y[j])
                        self.clean_buckets_x[j] = np.append(
                            self.buckets_x[i-1], self.clean_buckets_x[j])
                else:
                    if (slopes_vec[i] < 0):
                        j += 1  # new clean bucket
                        self.clean_buckets_y.append(self.buckets_y[i-1])
                        self.clean_buckets_x.append(self.buckets_x[i-1])
                    else:
                        self.clean_buckets_y[j] = np.append(
                            self.buckets_y[i-1], self.clean_buckets_y[j])
                        self.clean_buckets_x[j] = np.append(
                            self.buckets_x[i-1], self.clean_buckets_x[j])
            else:
                self.clean_buckets_y[j] = np.append(
                    self.buckets_y[i-1], self.clean_buckets_y[j])
                self.clean_buckets_x[j] = np.append(
                    self.buckets_x[i-1], self.clean_buckets_x[j])
            i -= 1

        self.clean_buckets_y = list(reversed(self.clean_buckets_y))
        self.clean_buckets_x = list(reversed(self.clean_buckets_x))

    def _final_test(self, slopes_vec_clean, var_vec_clean):
        """Second test for previously joined segments"""

        Z_critical = norm.ppf(1-self.max_p_value)  # test critical value

        first_segment = True
        i = len(self.clean_buckets_y)-1  # start by the most recent segment.
        if i == 0:
            if self.verbose:
                print("No slope change found.")
        while i > 0:
            if (var_vec_clean[i] + var_vec_clean[i-1]) == 0.0:
                if (slopes_vec_clean[i-1]-slopes_vec_clean[i]) == 0.0:
                    Z = 0
                else:
                    Z = 9e999
            else:
                Z = np.abs((slopes_vec_clean[i-1]-slopes_vec_clean[i]) /
                           np.sqrt(var_vec_clean[i] + var_vec_clean[i-1]))
            if Z >= Z_critical:
                if first_segment:
                    if self.verbose:
                        print("Z = {} >= Z_critical = {}".format(
                            np.round(Z, 4), np.round(Z_critical, 4)))
                    self.maturity = len(self.clean_buckets_y[i])
                    if self.verbose:
                        print("***\nSignificant maturity at {} "
                              "months\n***".format(self.maturity))
                    first_segment = False
            else:
                if first_segment:
                    if self.verbose:
                        print("Z = {} >= Z_critical = {}".format(
                            np.round(Z, 4), np.round(Z_critical, 4)))
                    self.maturity = len(self.clean_buckets_y[i])
                    if self.verbose:
                        print("***\nNon-significant maturity at {} months\n***"
                              "".format(self.maturity))
                    first_segment = False
            i -= 1

    def _piecewise_method(self):
        """Calculates the maturity with the piecewise method"""

        # derivation of the initial split points
        self.buckets_y, self.split_points = infer_change_points(
            self.y, self.x, self.beta, self.gamma, self.min_points)
        self.buckets_x = np.split(self.x, self.split_points)

        # statistical test and restictions
        (self.mult_reg, slopes_vec, var_vec, n_vec) = self._bucket_statistics(
            self.buckets_y, self.buckets_x)

        self._compliant_buckets(slopes_vec, var_vec)

        # get the maturity period
        (self.mult_reg_clean, slopes_vec_clean,
         var_vec_clean, n_vec_clean) = self._bucket_statistics(
            self.clean_buckets_y, self.clean_buckets_x)

        self._final_test(slopes_vec_clean, var_vec_clean)

        if self.verbose:
            self._plot_piecewise()

    def _plot_accumulative(self):
        """Plots the result of the accumulated method for the maturity."""

        if self.method != "accumulative":
            raise ValueError("Mehtod accumulative has not been executed")

        plt.figure(figsize=(15/2, 10/2))
        plt.plot(self.df_accum["accumulated_per"], color="#1464A5",
                 label="% accumulated bads")
        plt.bar(self.df_accum.index.values,
                self.df_accum["bads_perc"],
                color="#02A5A5", alpha=0.5, label="% total bads")
        plt.plot([0, np.max(self.y)], [0.6, 0.6], "-.", c="#DA3851",
                 linewidth=1)
        plt.plot([0, np.max(self.y)], [0.8, 0.8], "-.", c="#DA3851",
                 linewidth=1)
        plt.annotate("{} months".format(self.accum_06),
                     (self.accum_06, 0.55), color="#DA3851")
        plt.annotate("{} months".format(self.accum_08),
                     (self.accum_08, 0.75), color="#DA3851")
        plt.title("Accumulated defaults")
        plt.xlabel("months to default")
        plt.ylabel("defaults (%)")
        plt.xlim(0, np.max(self.y))
        plt.legend()
        plt.show()
        plt.close()

    def _plot_piecewise(self):
        """Plots the result of the piecewise method for the maturity."""

        if self.method != "piecewise":
            raise ValueError("Mehtod piecewise has not been executed")

        plt.figure(figsize=(20/2, 12/2))

        plt.plot(self.date_vec, self.y, "-", marker="o", c="#666666",
                 linewidth=1, label="data", markersize=3)
        plt.plot(self.date_vec, self.mult_reg, "--", c="#02A5A5",
                 label="pre-piecewise")
        plt.plot(self.date_vec, self.mult_reg_clean, "-", c="#1464A5",
                 label="final-piecewise")
        plt.xlim(min(self.date_vec), max(self.date_vec))
        plt.xlabel("formalization date")
        plt.ylabel("default rate")
        plt.legend()
        plt.show()

    def _run(self, df=None, y=None, x=None):
        if self.method == "piecewise":

            if df is not None:
                _date = pd.to_datetime(df[self.date], format=self.date_format)
                _date_no_day = [int(str(dt.year) + "{:02d}".format(dt.month))
                                for dt in _date]
                _date = pd.to_datetime(_date_no_day, format="%Y%m")
                _df = pd.DataFrame({self.date: _date,
                                    self.target: df[self.target]})

                _df_group = _df.groupby(self.date).agg(
                    {self.target: lambda x: np.mean(x)})

                self.y = _df_group[self.target].values
                self.x = np.arange(0, len(self.y))
                self.date_vec = _df_group.index.values
            else:
                self.y = y
                if x is None:
                    self.x = np.arange(0, len(self.y))
                else:
                    self.x = x
                self.date_vec = self.x

            if self.beta is None:
                self.beta = 2*np.log(len(self.y))

            self._piecewise_method()

        if self.method == "accumulative":

            self._n_samples = df.shape[0]

            df_months_to_bad = df[df[self.target] == 1][[
                self.default_date, self.date]].reset_index()

            s_bad = pd.to_datetime(df_months_to_bad[self.default_date],
                                   format=self.date_format)
            s_form = pd.to_datetime(df_months_to_bad[self.date],
                                    format=self.date_format)

            df_months_to_bad["months_diff"] = [s_bad[i].month - s_form[i].month
                + 12 * (s_bad[i].year - s_form[i].year)
                for i in range(len(s_bad))]

            self.y = df_months_to_bad["months_diff"].values
            self.x = None

            self._accumulative_method()
