import os
import pandas as pd
from pathlib import Path
import types, tqdm
import datetime as dt


class DataLoader:
    """
    Purpose
    * provide a path and return a generator of dfs to be loaded
    * process the df
    * method to save the df
    
    files are loaded in the order of file name
    """
    def __init__(self, path, suffix="parquet", **kwargs):
        self.i = 0
        
        p = Path(path)
        self.files = sorted(list(p.glob(f"*.{suffix}")))
        self.kwargs = kwargs
        
        try:
            self.load_fn = getattr(pd, f"read_{suffix}")
        except:
            raise ValueError(f"trying read_{suffix}, suppported read type - must be one from pd.read_* ")
    
        
    def __iter__(self):
        return self
    
    
    def __next__(self):
        return self.next()
    
    
    def next(self):
    
        if self.i < len(self.files):
            fpath = self.files[self.i]
            fname = fpath.stem
            df = self.load_fn(fpath, **self.kwargs)
            self.i += 1
            
            return fname, df
        
        raise StopIteration()
    
    
    def get_paths(self):
        return list(map(lambda fpath: str(fpath.absolute()), self.files))
    
    
    def get_full(self):
        dfs = [self.load_fn(fpath, **self.kwargs) 
               for fpath in self.files]
        return pd.concat(dfs, axis=0)

    

class DataDumper:
    
    def __init__(self, dir_path):
        """
        dir_path (str): path to directory
        """
        self.dir_path = dir_path
        os.makedirs(dir_path, exist_ok=True)
        
    
    def to_parquet(self, df, fname, suffix="parquet", **kwargs):
        """
        df (pd.DataFrame): dataframe to save
        fname (str): saved file will take form "{self.dir_path}/{fname}{suffix}"
        suffix (str): saved file will take form "{self.dir_path}/{fname}{suffix}"
        kwargs: to be passed into df.to_parquet(**kwargs)
        """
        fpath = os.path.join(self.dir_path, f"{fname}.{suffix}")
        df.to_parquet(fpath, **kwargs)
        
    
    def to_parquets(self, dfs, fname, append_timestamp=True, suffix="parquet", **kwargs):
        if not append_timestamp:
            raise NotImplementedError("Currently only supports append_timestamp=True")
            
        for df_ in tqdm.tqdm(dfs):
            dt_str = str(dt.datetime.now().timestamp())
            fname_ = f"{fname}_{dt_str}"
            self.to_parquet(df_, fname_, suffix=suffix, **kwargs)
    
    
    @staticmethod
    def get_timestamp_str(round=True):
        if round:
            return str(int(dt.datetime.now().timestamp()))
        return str(dt.datetime.now().timestamp())