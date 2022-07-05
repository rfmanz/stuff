"""
datagen.py

Data generator abstraction for training functions.
"""
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


class BaseDataGenerator:
    """Base class for all data generators.
    
    Working note: this class is very broad because data 
    can be sampled in all sorts of ways and we don't want 
    get stuck with a system that doesn't work for complex 
    models.
    """

    def __init__(self):
        raise NotImplementedError
        
    def __iter__(self):
        """Generator function.
        """
        raise NotImplementedError
        

class GroupKFoldGenerator(BaseDataGenerator):
    
    def __init__(self, df, n_splits, strategize_by, groupby, seed=12345):
        self.df = df
        self.n_splits = n_splits
        self.n = 0
        self.groupby = groupby
        self.seed = seed
        self.folds = self.get_splits(df)

    def get_splits(self, df):
        if self.seed:
            random.seed(self.seed)
        
        grp = df[self.groupby]
        grpkfold = GroupKFold(n_splits=self.n_splits)
        folds = grpkfold.split(df, groups=grp)
        return folds

    def __iter__(self):
        self.n = 0
        return self
    
    
    def __next__(self):
        if self.n <= self.n_splits:
            train_idx, test_idx = next(self.folds)
            df_train = self.df.iloc[train_idx]
            df_test = self.df.iloc[test_idx]
            
            self.n += 1
            return (df_train, df_test)
        else:
            raise StopIteration
            
    
    def __len__(self):
        return self.n_splits

    
def get_positive_label_weight(series):
    """
    Get scaling factor for binary classification task with imbalanced labels
    """
    count_pos = series.sum()
    count_neg = (~series).sum()
    scaling_factor = count_neg / count_pos
    return scaling_factor