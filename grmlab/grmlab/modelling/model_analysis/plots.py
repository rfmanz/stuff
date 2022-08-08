"""
Plots used in model analyzer.
"""

# Authors: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2019.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import gaussian_filter1d


class GaussianFilter(object):
    """Derivation of the distribution of a set of data values.

    Parameters
    ----------
    data : list of float
        Data values to calculate the distribution.

    nbins : int
        Number of bins for the histogram. If no value is given, it
        calculates the number of bins with the Freedman-Diaconis rule.

    sigma : int (default=2)
        Standard deviation of the Gaussian kernel. The lower the sigma,
        the higher the over-fitting.
    """
    def __init__(self, data=None, nbins=None, sigma=2):
        self._nbins = nbins

        self.x = None
        self.y = None
        self._y_data = None
        self._x_data = None

        if data is not None and len(data) > 1:
            if self._nbins is None:
                # Freedman-Diaconis rule for nbins
                h = (2 * (np.percentile(data, 75) - np.percentile(data, 25)) *
                     np.power(len(data), -1/3))
                if h > 0:
                    self._nbins = int(max(round((max(data) - min(data))/h), 1))
                else:
                    self._nbins = 1

            self._binning(data)
            self._smooth(self._x_data, self._y_data, sigma)

    def _binning(self, data):
        """Calculates the histogram"""
        self._y_data, bins = np.histogram(data, bins=self._nbins,
                                          density=False)
        # middle points of each bin
        self._x_data = [(bins[i]+bins[i+1]) / 2 for i in range(len(bins) - 1)]

    def _smooth(self, x_data, y_data, sigma=2):
        """Calculates the gaussian kernel"""
        self.x = gaussian_filter1d(np.array(x_data), sigma)
        self.y = gaussian_filter1d(np.array(y_data), sigma)

    def plot(self, name=""):
        """Plot histogram with Gaussian Kernel.

        Parameters
        ----------
        name : str
            Name of the variable. It will be shown in the x-label and in the
            title.
        """
        width_bars = 0.8*(min(self._x_data)-max(self._x_data))/self._nbins
        plt.figure(figsize=(15/2, 10/2))
        plt.bar(self._x_data, self._y_data, color="#2dcccd",
                width=width_bars, alpha=0.3, label="histogran")
        plt.plot(self.x, self.y, "--", color="#1464A5", label="kernel")
        plt.legend()
        plt.xlabel(name)
        plt.ylabel("Elements per bin")
        plt.title("{} Distribution".format(name))
        plt.show()
        plt.close()


def plot_roc(fpr, tpr):
    """Plots the ROC curve.

    Parameters
    ----------
    tpr : list of float
        True positive rate vector at various threshold settings.

    fpr : list of float
        False positive rate vector at various threshold settings.
    """
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0.0, 0.0, 1.0], [0.0, 1.0, 1.0], 'g--', label="perfect")
    plt.plot([0, 1], [0, 1], "r--", label="random")
    plt.legend()
    plt.xlabel("%FPR")
    plt.ylabel("%TPR")
    plt.title("ROC Curve")
    plt.show()
    plt.close()


def plot_roc_comp(name1, fpr1, tpr1, name2, fpr2, tpr2):
    """Plots the ROC curve.

    Parameters
    ----------
    tpr : list of float
        True positive rate vector at various threshold settings.

    fpr : list of float
        False positive rate vector at various threshold settings.
    """
    plt.plot(fpr1, tpr1, label="ROC curve " + name1)
    plt.plot(fpr2, tpr2, label="ROC curve " + name2)
    plt.plot([0.0, 0.0, 1.0], [0.0, 1.0, 1.0], 'g--', label="perfect")
    plt.plot([0, 1], [0, 1], "r--", label="random")
    plt.legend()
    plt.xlabel("%FPR")
    plt.ylabel("%TPR")
    plt.title("ROC Curve")
    plt.show()
    plt.close()


def plot_correlation_matrix(names, max_corr, correlations):
    """Plots the correlation matrix as a heat map

    Parameters
    ----------
    names : list of strings
        Names of the variables.

    max_corr : float
        Max correlation in the color bar.

    correlations : np.array()
        Correlation matrix.
    """

    # Exclude diagonal values
    mask = np.zeros_like(correlations, dtype=np.bool)
    mask[np.diag_indices_from(mask)] = True

    # max val based on limit set in feature selection.
    corr = correlations - np.diag(np.diag(correlations))
    max_val = max(max([elt for elt in corr[mask]]), max_corr)

    # Set background color / chart style
    sns.set_style(style='white')

    # Set up  matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Colo palette
    cmap = sns.color_palette("Blues", n_colors=40)

    # Draw correlation plot
    sns.heatmap(correlations, mask=mask, cmap=cmap,
                vmax=max_val, vmin=0, square=True,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

    plt.yticks(np.arange(0, len(names)), names)
    plt.yticks(rotation=0, va="top")
    plt.xticks(np.arange(0, len(names)), names)
    plt.xticks(rotation=90, ha="left")
    plt.title("Correlation Matrix")
    plt.show()
    plt.close()


def plot_multiclass_confusion_matrix(names_x, names_y, conf_matrix):

    # Set background color / chart style
    sns.set_style(style='white')

    # Set up  matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Colo palette
    cmap = sns.color_palette("Blues", n_colors=40)

    # Draw correlation plot
    sns.heatmap(conf_matrix, cmap=cmap, vmin=0, square=True,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

    plt.yticks(np.arange(0, len(names_y)), names_y)
    plt.yticks(rotation=0, va="top")
    plt.xticks(np.arange(0, len(names_x)), names_x)
    plt.xticks(rotation=90, ha="left")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted values")
    plt.ylabel("True values")
    plt.show()
    plt.close()


def plot_sector_error(names_x, names_y, error_matrix):

    # Set background color / chart style
    sns.set_style(style='white')

    # Set up  matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Colo palette
    cmap = sns.color_palette("Blues", n_colors=40)

    # Draw correlation plot
    sns.heatmap(error_matrix, cmap=cmap, vmin=0, square=True,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

    plt.yticks(np.arange(0, len(names_x)), names_x)
    plt.yticks(rotation=0, va="top")
    plt.xticks(np.arange(0, len(names_y)), names_y)
    plt.xticks(rotation=90, ha="left")
    plt.title("Relative error distribution by range")
    plt.xlabel("Relative Error")
    plt.ylabel("True values range")

    plt.show()
    plt.close()
