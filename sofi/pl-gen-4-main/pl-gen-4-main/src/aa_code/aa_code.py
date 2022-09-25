import os
from abc import ABCMeta, abstractmethod
from smart_open import open    
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
from .pdp import (
    pdp_w_impact_table, 
    plot_pdp_w_impact, 
    build_aa_code_pdp
)


class AACodeBase(metaclass=ABCMeta):
    
    def __init__(self, *args, **kwargs):
        for k in kwargs:
            setattr(k, kwargs[k])
            
    
    @abstractmethod
    def fit(self, X, y=None, model=None, **kwargs):
        """
        build necessary components for AA Code generation
        
        e.g. for PDP AA code with LGBM + MC, produce the pdp object here.
             for SHAP based method, produce shap explaner.
        """
        raise NotImplementedError
        
        
    @abstractmethod
    def transform(self, X, **kwargs):
        """
        produce impact output here.
        
        this would be the only function being called at inference time.
        """
        raise NotImplemnetedError
        
    
    def save(self, path):
        """
        save necessary state dicts to pkl for later loads
        """
        raise NotImplementedError
        
    
    def load(self, path):
        """
        load previously computed state dicts from a pkl file
        """
        raise NotImplementedError
        
        

class AACodePDP(AACodeBase):
    """
    Generate AA Code based on PDP
    
    Procedure:
    1. normalize the risk model scores to a common range of values 0 - 100
    2. subtract the normalized value from 100 to convert to a weakness score
        * higher the weakness, e.g. distance from idea, 
          the more impactful the score is contributing to the decline
    3. return the reasons based on the weakness scores
    
    As of 12/2021, modules takes in LGBM booster and data to generate 
    list of impact features as a pandas df
    """

    def __init__(self, model, aa_df, features, **kwargs):
        self.model = model
        self.aa_df = aa_df
        self.features = features
        
    
    def fit(self, df, num_cuts=100,
            include_missing=False, **kwargs):
        self.pdp_dict = pdp_w_impact_table(self.model, df, self.features, 
                                           num_cuts=num_cuts, 
                                           include_missing=include_missing)
    
    
    def plot(self, max_n_ticks=10, ncols=3, figsize=None, **kwargs):
        assert(hasattr(self, "pdp_dict"))
        fig, axs = plot_pdp_w_impact(self.pdp_dict, 
                                     self.features, 
                                     max_n_ticks=max_n_ticks,
                                     ncols=ncols,
                                     figsize=figsize,
                                     **kwargs)
        return fig, axs
    
    
    def transform(self, df, inq_fts=[], 
                  aa_code_valid_col="AA_code_valid",
                  aa_code_special_col="AA_code_special_value",
                  missing_idx=None):
        assert len(df) == 1  # right now we inference one point at a time
        impact_df, top_aa = build_aa_code_pdp(df, 
            self.pdp_dict, self.aa_df, 
            inq_fts=inq_fts,
            aa_code_valid_col=aa_code_valid_col,
            aa_code_special_col=aa_code_special_col,
            missing_idx=missing_idx)
        return top_aa
    
    
    def save(self, dir_path):
        """
        save the state dictionary of this object to directory provided
        """
        os.makedirs(dir_path, exist_ok=True)
        
        with open(os.path.join(dir_path, "model.pkl"), "wb") as f:
            pkl.dump(self.model, f)
            
        with open(os.path.join(dir_path, "features.pkl"), "wb") as f:
            pkl.dump(self.features, f)
            
        self.aa_df.to_csv(os.path.join(dir_path, "aa_df.csv"))
        
        if hasattr(self, "pdp_dict"):
            with open(os.path.join(dir_path, "pdp_dict.pkl"), "wb") as f:
                pkl.dump(self.pdp_dict, f)
            
        print(f"objects saved at {dir_path}")
    
    
    @classmethod
    def load(cls, dir_path):
        """
        load AA_Code class from directory
        
        @params dir_path: str
            directory path to AA code objects
            - model.pkl
            - features.pkl
            - aa_df.pkl
            - pdp_dict.pkl
        
        @returns AACode class
        """
        print("loading back objects")
        
        with open(os.path.join(dir_path, "model.pkl"), "rb") as f:
            model = pkl.load(f)
        
        with open(os.path.join(dir_path, "features.pkl"), "rb") as f:
            features = pkl.load(f)
            
        aa_df = pd.read_csv(os.path.join(dir_path, "aa_df.csv"), index_col=0)
        
        aa_code = cls(model, aa_df, features)
        
        with open(os.path.join(dir_path, "pdp_dict.pkl"), "rb") as f:
            aa_code.pdp_dict = pkl.load(f)
            
        return aa_code