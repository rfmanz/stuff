# from src.feature.transform_v1 import transform
from src.feature.transform_v2 import transform

class FeatureEngineering:
    def __init__(self, config, prefix_in, prefix_out, logger):
        self.config = config
        self.prefix_in = prefix_in
        self.prefix_out = prefix_out
        self.logger = logger
        self.dup_token = "_<DUP>_"

    def run(self, dfs):
        dfs, self.config = self._run(dfs)
        return dfs, self.config

    def _run(self, dfs):
        # df = transactions_df
        
        result_dfs = {}
        for fname, df in dfs.items():
            df, content = transform(df)
            result_dfs[fname] = df
            
        return result_dfs, self.config
        

