import sys, os, shap, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rdsutils.plot as rdsplot
from tqdm import tqdm

######################################
#               General
######################################


def get_grid_plots(plot_fn, context, features, ncols=6, figsize=None, wspace=0.4, **kwargs):
    """
    build the grid plots with plot_fn and given context
    
    plot_fn(feature, context, ax), 
    plot_fn will use info in context to plot on ax, for provided feature
    """
    nrows = math.ceil(len(features) / ncols)
    
    if figsize is None:
        figsize = (ncols * 6, nrows * 6)
        
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for feature, ax in tqdm(zip(features, axs.flatten())):
        plot_fn(feature, context, ax, **kwargs)
    
    plt.subplots_adjust(wspace=wspace)
    return fig


######################################
#               SHAP
######################################

def get_shap_dependence(df, shap_values, features, ncols=6, figsize=None, **kwargs):
    """
    Build the partial dependence plot for a set of models and features.
    """
    nrows = math.ceil(len(features) / ncols)

    if figsize is None:
        figsize = (ncols * 6, nrows * 6)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for feature, ax in zip(features, axs.flatten()):
        shap.dependence_plot(feature, shap_values, df, 
                             ax=ax, show=False, **kwargs)
        rlim = df[feature].quantile(0.98)
        llim = df[feature].quantile(0.02) - ((rlim - df[feature].quantile(0.02)) / 12)
            
        if rlim < np.inf and llim > -np.inf:
            ax.set_xlim(left=llim, right=rlim)
        
    return fig


def plot_shap_dependence(feature, context, ax, **kwargs):
    shap_values = context["shap_values"]
    df = context["df"]
    shap.dependence_plot(feature, shap_values, df, 
                         ax=ax, show=False, **kwargs)
    rlim = df[feature].quantile(0.98)
    llim = df[feature].quantile(0.02) - ((rlim - df[feature].quantile(0.02)) / 12)

    if rlim < np.inf and llim > -np.inf:
        ax.set_xlim(left=llim, right=rlim)
    
    return ax

def plot_shap_interaction(feature, context, ax, interaction_index, **kwargs):
    """
    plot how given feature interact with the interaction_feature
    """
    shap_values = context["shap_values"]
    df = context["df"]
    shap.dependence_plot(interaction_index, shap_values, df, ax=ax, 
                         show=False, interaction_index=feature, **kwargs)
    rlim = df[interaction_index].quantile(0.98)
    llim = df[interaction_index].quantile(0.02) - ((rlim - df[interaction_index].quantile(0.02)) / 12)
    
    if rlim < np.inf and llim > -np.inf:
        ax.set_xlim(left=llim, right=rlim)
    
    return ax


######################################
#            Correlation
######################################


def get_corr_w_target(df, features, target_col="target"):
    corrs = []
    for f in features:
        c = df[[f, target_col]].astype(float).corr().iloc[1,0]
        corrs.append((f, c))
        
    corrs = pd.DataFrame(corrs)
    corrs.columns = ["feature", "corr"]
    return corrs


######################################
#                 WOE
######################################

def plot_woe(fname, woe_dict, ax=None):
    if ax is None:
        fig = plt.figure()
    x = woe_dict[fname]["min"]
    y = woe_dict[fname]["woe"]
    ax.plot(x, y)
    ax.set_title(f"{fname}")
    
def get_woe_plots(df, woe_dict, features, ncols=6, figsize=None, **kwargs):
    """
    Build the partial dependence plot for a set of models and features.
    """
    nrows = math.ceil(len(features) / ncols)

    if figsize is None:
        figsize = (ncols * 6, nrows * 6)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for feature, ax in zip(features, axs.flatten()):
        plot_woe(feature, woe_dict, ax=ax)        
    return fig


######################################
#                 PDP
######################################


def partial_dependency(model, df, feature, features, 
                       n_steps=10, sample_size=None):
    """
    Calculate partial dependency of a feature given a model.
    """
    if sample_size:
        d = df.sample(sample_size).copy()
    else:
        d = df.copy()
    grid = np.linspace(df[feature].quantile(0.001),
                       df[feature].quantile(.995),
                       n_steps)
    preds = []

    for x in grid:
        d[feature] = x
        y = np.average(model.predict(d[features]))
        preds.append([x, y])
    return np.array(preds).T[0], np.array(preds).T[1]


def partial_dependency_plot_cv(ax, models, df, 
                               feature, features, n_steps=10, 
                               sample_size=None, ylab=None):
    """
    Return partial dependence plot for a feature on a set of models.
    """
    d = df.copy()

    partial_dependencies = []
    y_mean = np.array([0] * n_steps)
    x_mean = []

    y_min = np.inf
    y_max = -np.inf
    d[d[feature] == np.inf] = np.nan #edge case

    for model in models:
        x, y = partial_dependency(model, d, feature, 
                                  features, n_steps=n_steps, 
                                  sample_size=sample_size)
        
        y_min = min(y_min, min(y))
        y_max = max(y_max, max(y))
        
        y_mean = y_mean + (y / len(models))
        x_mean = x
        partial_dependencies.append([x, y])

        for x, y in partial_dependencies:
            ax.plot(x, y, '-', linewidth=1.4, alpha=0.6)

    ax.plot(x_mean, y_mean, '-', color = 'red', linewidth = 2.5)
    ax.set_xlim(d[feature].quantile(0.001), d[feature].quantile(0.995))
    ax.set_ylim(y_min*0.99, y_max*1.01)
    ax.set_xlabel(feature, fontsize = 10)
    if ylab:
        ax.set_ylabel(ylab, fontsize = 12)
                
            
def get_pdp(df, features, models, ncols=6, 
            figsize=None, sample_size=None):
    """
    Build the partial dependence plot for a set of models and features.
    """
    if type(models) is not list:
        models = [models]

    nrows = math.ceil(len(features) / ncols)

    if figsize is None:
        figsize = (ncols * 6, nrows * 6)

    fig, axs = plt.subplots(nrows=nrows, 
                            ncols=ncols, 
                            figsize=figsize)
    for feature, ax in tqdm(zip(features, axs.flatten())):
        try:
            partial_dependency_plot_cv(ax, models, df, 
                                       feature, features, 
                                       sample_size=sample_size)
        except:
            continue
    return fig