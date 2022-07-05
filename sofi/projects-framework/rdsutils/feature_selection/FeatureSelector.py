"""
Authors:
George Xu
Bo Shao
Thomas Boser
"""
import gc, os, sys
import time
import numpy as np
import seaborn as sns
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import pickle as pkl
from smart_open import open

# utilities
from itertools import chain
from pathlib import Path

# custom functions
from .boruta import Boruta
from .woe import WOE_Transform
from .utils import display_feature_importance, display_corr_matrix
from .get_stats import get_missing_, get_single_unique_
from .get_woe import get_woe_
from .get_collinear import get_collinear_, get_collinear_features_to_drop_
from .lgb_importance import get_default_lgb_estimator_
from .lgb_importance import get_lgb_importance_



class FeatureSelector():
    
    def __init__(self, data, label_cols=None, 
                 feature_cols=None, id_col=None):
        """
        Feature Selection class
        
        TODO: add docs
        """
        
        # Dataset and optional training labels
         # should not run into memory issue since we are not creating another copy
            
        self.data = data
        self.features = feature_cols
        self.label_cols = label_cols
        self.id_col = id_col
        
        if label_cols is None:
            print('No labels provided. Feature importance based methods are not available.')
        
        self.base_features = list(data.columns)
        self.one_hot_features = None
        
        # Dataframes recording information about features to remove
        self.record_missing = None
        self.record_single_unique = None
        self.record_collinear = None
        self.record_feature_importance = None
        self.record_low_importance = None
        self.record_boruta_importance = None
        self.record_iv = None
        
        self.missing_stats = None
        self.unique_stats = None
        self.corr_matrix = None
        self.feature_importances_folds = None
        self.iv = None
        
        # Dictionary to hold removal operations
        self.ops = {}
        
        
    def get_missing(self, missing_threshold):
        self.missing_threshold = missing_threshold
        missing_df, to_drop = get_missing_(self.data,
                                             self.features,
                                             self.missing_threshold)
        self.record_missing = missing_df
        self.ops['missing'] = to_drop
        print(f'{len(self.record_missing)} features with greater than \
                {round(self.missing_threshold, 2)} missing values')
        return missing_df
        
    
    def get_single_unique(self, count_nan=False):
        single_unique_df, to_drop = get_single_unique_(self.data,
                                                       self.features, 
                                                       count_nan)
        self.record_single_unique = single_unique_df
        self.ops['single_unique'] = to_drop
        print(f'{len(self.record_single_unique)} features with a single unique value.')
        return single_unique_df
        
    
    def get_default_lgb_estimator(self, task, n_estimators=1000,
                                  class_weight=None, **kwargs):
        return get_default_lgb_estimator_(task, n_estimators,
                                          class_weight, **kwargs)
        
    
    def get_lgb_importance(self, task, model=None, eval_metric=None,
                           n_iterations=5, early_stopping_rounds=100, 
                           cat_features=None, n_estimators=1000,
                           class_weight=None, group_col_name=None, **kwargs):
        """ Get LightGBM importance 
        @params task: What type of task: ['classification', 'refression']
        @params eval_metric: lightgbm eval metrics, e.g. auc, logloss, mse
        @params n_iterations: number of iterations to run
        @params early_stopping_rounds: num rounds to early stop
        @params cat_features: list of categorical features
        @params group_col_name: id column used to do GroupKFold on
        
        @returns feature importances df
        """
        fimp_folds, fimp = get_lgb_importance_(self.data, 
                                               self.features,
                                               task,
                                               self.label_cols,
                                               model=model,
                                               eval_metric=eval_metric,
                                               n_iterations=n_iterations,
                                               early_stopping_rounds=early_stopping_rounds,
                                               cat_features=cat_features,
                                               n_estimators=n_estimators,
                                               class_weight=class_weight,
                                               group_col_name=group_col_name,
                                               **kwargs)
        self.feature_importance_folds = fimp_folds
        self.record_feature_importance = fimp
        return fimp
    
    
    def get_low_feature_importance(self, threshold=0):
        """
        @params threshold: threshold for features
        @returns low_fimp: df with containing low feature importance
        """
        if self.record_feature_importance is None:
            print('please run get_lgb_importance first')
            sys.exit(1)
        fimp = self.record_feature_importance
        low_fimp = fimp[fimp.importance < threshold]
        self.record_low_importance = low_fimp
        
        low_fimp_features = low_fimp.feature.tolist()
        self.ops['low_importance'] = low_fimp_features
        return low_fimp

    
    def plot_feature_importance(self, max_n_features=20, figsize=(9,7)):
        fimps = self.feature_importance_folds
        fig, ax = display_feature_importance(fimps.feature,
                                         fimps.importance, 
                                         max_n_features=max_n_features,
                                         figsize=figsize)
        plt.show()
        return fig, ax
    
    
    
    def get_boruta_importance(self, estimator, features=None, max_iter=20, 
                              min_features=1, thresh=0.25, 
                              drop_at=None, random_state=None, verbose=0):
        """ Wrapper function for rdsutils.boruta.Boruta
        check it out for more details
        
        Please take care of categorical encoding outside of calling this fn
        - when defining the estimator or processing dataset
        
        For conveniance, call get_default_lgb_estimator() for a default lgbm
        
        @param estimator
        @param max_iter
        @param min_features
        @param thresh
        @param drop_at
        @param random_state
        @param verbose
        
        @returns boruta importance df
        """
        
        boruta = Boruta(estimator, max_iter=max_iter, 
                        min_features=min_features,
                        thresh=thresh, drop_at=drop_at, 
                        random_state=random_state,
                        verbose=verbose)
        
        features = features or self.features
        boruta.fit(self.data[features].values, 
                   self.data[self.label_cols].values) 
        
        result = {'feature': features,
                  'score': boruta.scores,
                  'mean_importance': np.mean(boruta.imps, axis=0)}
        bimp = pd.DataFrame(result).sort_values(by=['score', 
                                                    'mean_importance'],
                                                ascending=False)
        bimp['mean_importance'] = (bimp['mean_importance'] 
                                   / bimp['mean_importance'].sum()
                                   * 100)
        self.record_boruta_importance = bimp
        return bimp
    
    
    def get_woe(self, method='tree', num_features=None, num_bin_start=40, 
                min_samples_leaf=500, min_iv=0.01, display=0, 
                woe_fit_params={}):
        """ WOE wrapper for rdsutils.woe 
        
        For detailed usage or customization, checkout the original repo
        """
        self.woe = get_woe_(self.data, self.features,
                            self.label_cols, method=method,
                            num_features=num_features,
                            num_bin_start=num_bin_start,
                            min_samples_leaf=min_samples_leaf,
                            min_iv=min_iv,
                            display=display,
                            woe_fit_params=woe_fit_params)
        return self.woe
        
        
    def get_iv(self, method='tree', num_features=None, num_bin_start=20, 
               min_samples_leaf=100, min_iv=0.01, display=0, **kwargs):
        """
        Compute information value
        
        @returns record_iv: dataframe ordered by information value
        """
        if num_features is not None or not hasattr(self, 'woe'):
            _ = self.get_woe(method=method,num_features=num_features, 
                             num_bin_start=num_bin_start, 
                             min_samples_leaf=min_samples_leaf, 
                             min_iv=min_iv, display=display, **kwargs)
        self.record_iv = self.woe.get_iv().sort_values(by=['iv'], 
                                                       ascending=False)
        print("IV produced")
        return self.record_iv
        
    
    def get_collinear(self, corr_lowerbound=-1, corr_upperbound=1,
                      num_features=None):
        """
        @returns collinear_mtx: collinear matrix containing high values
        @returns to_drop: high correlation pairs
        """
        print(f"correlation lowerbound: {corr_lowerbound}\n"
              f"correlation upperbound: {corr_upperbound}")
        self.acceptable_corr_range = (corr_lowerbound, corr_upperbound)
        collinear_mtx, high_corr_pairs = get_collinear_(self.data,
                                                self.features,
                                                corr_lowerbound=corr_lowerbound,
                                                corr_upperbound=corr_upperbound,
                                                num_features=num_features)
        self.ops['collinear'] = high_corr_pairs
        self.record_collinear = collinear_mtx
        self.record_collinear_pairs = high_corr_pairs
        print(f"{len(high_corr_pairs)} features have correlation"
              f"beyond bounds provided")
        return collinear_mtx, high_corr_pairs
    
    
    def plot_collinear(self, figsize=(10,6), title=None, cmap='coolwarm', **kwargs):
        fig, ax = display_corr_matrix(self.record_collinear, 
                                      figsize=figsize,
                                      title=title,
                                      cmap=cmap, 
                                      **kwargs)
        plt.show()
        return fig, ax
    
    
    def get_collinear_features_to_drop(self, removal_order):
        """
        @params removal_order: list of features to remove 
                - each feature index reflecting its importance
                - features come early in the list are less important
                - e.g. ['a', 'b', 'c'],  imp('c') = 2 > imp('a') = 0
        """
        return get_collinear_features_to_drop_(self.record_collinear_pairs, 
                                               removal_order)
    
    
    def save(self, save_path):
        """
        save the state dictionary
        """
        
        
        p = Path(save_path)
        os.makedirs(p.parent, exist_ok=True)
        
        # remove elements with None values
        context = {k: v for k, v in vars(self).items() if v is not None}
        
        # do not store raw data again.
        _ = context.pop('data')
        
        with open(save_path, 'wb') as f:
            pkl.dump(context, f)
            print("state dictionary saved!")
            
            
    def save_state_dict(self, save_path):
        """
        same as save
        """
        self.save(save_path)
        
        
    def load_state_dict(self, save_path):
        """
        Replicate previously created fsel objects.
        - so can continue working or use the plotting functionalities.
        
        1. initialize the FeatureSelector object with the same dataset
        2. assign state values to the attributes
        """
        state_dict = pkl.load(open(save_path, 'rb'))
        for k in state_dict:
            setattr(self, k, state_dict[k])
        print("state dictionary loaded!")
