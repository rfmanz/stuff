# General Class to unify data pipeline

import os, sys
import argparse, datetime, gc, time, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Sequence, List
import src.utils as utils
import math

# rdsutils.aws.upload_s3
# rdsutils.query.query_postgres
# mdsutils.AthenaClient
# mdsutils.S3Object


class Dataloader:
    def __init__(
        self,
        config_path: str = "config.json",
        debug: bool = False,
    ):
        """
        An extendable data loader class for Money Transactions data
        ---

        @params config_path: str
            path to config file from current working directory (os.getcwd())
        @params debug: bool
            whether to enter debug mode, must provide "debug_ids_path" in config.json
        """
        self.config_path = config_path
        self.config = self.get_config(config_path)
        self.debug = debug

        self.src_path = os.path.dirname(os.path.realpath(__file__))
        self.base_path = self.config["base_path"]
        debug_id_path = self.config["debug_ids_path"]
        self.log_path = self.config["log_path"]

        self._prefixs, self._names = set(), set()

        if debug:
            assert self.config["debug_ids_path"] is not None
            self.debug_id = pd.read_csv(debug_id_path)

        # all created object will have the same id
        self.time_id = utils.get_timestamp_str()
        self.config["time_id"] = self.time_id
        self.logger = utils.get_file_stdout_logger(
            self.log_path,
            f"log_{self.time_id}",
            name=__name__,
            append_dir_timestamp=False,
        )
        self.set_config(self.config)

    def query(
        self,
        prefix: str,
        by_id: pd.DataFrame = None,
        debug: bool = False,
        source: str = None,
    ):
        """
        General Query function for pipeline
        - supports querying from dw as of now. Extend to its separate class later

        @params prefix: str
            folder name for saving data
        @params by_id: pd.DataFrame
            if provided, only run queries on provided ids.
            to use this functionality, please provide:
                1. sql query with PLACEHOLDER to substituting into ids
                    e.g. src/sql-by-id/*.sql files
                2. update config.json -> sql_query_files -> channel -> "query_by_id"
                    to link to the corresponding sql queries
                    e.g. config.json
                3. construct a dataframe with columns being the corresponding ids.
                    each column title corresponding to a PLACEHOLDER to be replaced.
                    e.g. debug_ids.csv

                    convention: columns = f"<{id_name.upper()}>":
                    - <BORROWER_ID>, <USER_ID>, <BUSINESS_ACCOUNT_NUMBER>
        @params debug: if debug is True, load the debug_ids into by_id
        @params source: query specific data source
        """
        if by_id is not None:
            self._query_dw(self.base_path, prefix=prefix, by_id=by_id, source=source)
        elif debug:
            self._query_dw(
                self.base_path, prefix=prefix, by_id=self.debug_id, source=source
            )
        else:
            self._query_dw(self.base_path, prefix=prefix, by_id=None, source=source)
            
        self.config["data_pull_date"] = pd.datetime.today().strftime('%Y-%m-%d')
        self.set_config(self.config)

    def _query_dw(
        self, base_path: str, prefix: str, by_id: pd.DataFrame = None, source=None
    ):
        # record keeping
        self._prefixs.add(prefix)
        for name, qinf in self.config["sql_query_files"].items():

            if source is not None:
                # if source is provided, only query the appointed ont
                if source != name:
                    continue

            # record keeping
            self._names.add(name)

            # query_fn = query filename
            query_fn = qinf["query"] if by_id is None else qinf["query_by_id"]
            query_file = os.path.join(self.src_path, query_fn)

            with open(query_file) as f:
                query = f.read()
            # modify query here to add conditions

            self.logger.info(f"Querying from {query_file}")
            if by_id is not None:
                for col in by_id.columns:
                    ids = str(tuple(by_id[col].astype(str).values))
                    query = query.replace(col, ids)

            self.logger.info(f"Running query {name}.")

            df = utils.query_postgres(
                query=query, port=qinf["port"], database=qinf["database"]
            )
            self.save_dataframes(
                {name: df}, base_path, prefix, self.time_id, include_subdir=True
            )

    def save_dataframes(
        self,
        dataframes: pd.DataFrame,
        base_path: str,
        prefix: str = "",
        timestamp_str: str = None,
        include_subdir: bool = False,
    ):
        """
        Takes a dictionary of dataframes with format: {name: dataframe, ...}
        Saves them as base_path/prefix-path/name-timestamp.parquet
        
        TODO:
        - add chunk functionality
        - if provided chunking parameters, save data by chunks.
        - Q: what is the best way to provide the chunking logic? iterator?
        """

        self.config = utils._save_dataframes(
            self.config, dataframes, base_path, prefix, timestamp_str, include_subdir
        )
        self.set_config(self.config)

    def load_dataframe(self, prefix, name, base_path="data"):
        """
        Load individual dataframe from path in config file
        """
        return utils._load_dataframe(self.config, prefix, name, base_path)

    def load_dataframes(self, prefix, base_path="data", use_dask=False):
        """
        Load all dataframes from paths in config
        
        TODO:
        - add chunk functionality
        - if provide by some logic, we load and dump by chunks
        """
        return utils._load_dataframes(self.config, prefix, base_path, use_dask)

    def clean(self, style: str):
        """
        iteratively remove all data files that is not registered in config
        then remove empty directories

        @param style: str
            remove_current: remove the data included in the config
            remove_non_current: remove all but data listed in config
        """
        if style not in ["remove_current", "remove_non_current"]:
            raise ValueError(
                'Please choose between "remove_non_current" and "remove_current"'
            )

        # files currently registered in config.json
        current_files = []
        folders = []
        base_path = self.base_path
        for prefix in self.config["data"]:
            name_path_pairs = self.config["data"][prefix]
            for name, path in name_path_pairs.items():
                data_path = os.path.join(base_path, path)
                current_files.append(data_path)

        # all files under base_path dir
        data_files = []
        for path, subdirs, files in os.walk(self.base_path):
            folders.append(path)
            for file in files:
                file_path = os.path.join(path, file)
                data_files.append(file_path)

        if style == "remove_current":
            to_remove = current_files
        elif style == "remove_non_current":
            to_remove = list(set(data_files) - set(current_files))

        for file in tqdm(to_remove):
            os.remove(file)

        # remove all empty dirs in base_path
        for folder in folders:
            is_empty = len(os.listdir(folder)) == 0
            if is_empty:
                os.rmdir(folder)

        self.logger.info("Cleaned up!")

    def get_config(self, config_path=None):
        """
        default config_path can be obtained using:

        src_base = os.path.dirname(os.path.realpath(__file__))
        config_file_path = os.path.abspath(os.path.join(src_base, ../config.json))
        """

        if config_path is not None:
            with open(config_path, "r") as f:
                self.config = json.load(f)
            return self.config

        elif self.config is not None:
            return self.config

        raise ValueError("please provide config_path")

    def get_prefix(self):
        return list(self._prefixs)

    def get_source_names(self):
        return list(self._names)

    def set_config(self, config_file: dict):
        with open(self.config_path, "w") as f:
            json.dump(config_file, f, indent=4)

    def pipeline(self, steps: List = []):
        """
        @params steps: list of functions
            to apply in sequential manner
            if any error exist, terminate the process
        """
        for fn in steps:
            try:
                fn()
            except:
                self.logger.error(
                    f"pipeline failed at stage: {fn.__name__}\n"
                    f"check out {self.log_path} for log."
                )
                sys.exit()

    def process(
        self, prefix_in: str = "raw", prefix_out: str = "processed", debug: bool = False
    ):
        # load everything from previous step to dfs
        # pass dfs to processor
        # get dfs back from processor
        # save dfs
        # question: should we use dask???

        from src.process import Processor

        processor = Processor(self.config, prefix_in, prefix_out, self.logger)
        self.config = processor.run()
        self.set_config(self.config)


    def join(
        self,
        prefix_in: str = "processed",
        prefix_out: str = "joined",
        debug: bool = False,
        use_dask = False,
    ):
        # equally sliced method using sklearn
        from sklearn.model_selection import GroupKFold
        from src.join import Joiner
        
        chunk_size = self.config["chunk_size"]
        dfs = self.load_dataframes(prefix_in, self.base_path, use_dask=use_dask)
        joiner = Joiner(self.config, prefix_in, prefix_out, self.logger)
        assert("business_account_number" in self.config["meta_cols"])
        
        tdf = dfs["transactions"]
        n = math.ceil(len(tdf) / chunk_size)
        gkf = GroupKFold(n_splits=n)
        ids = []
        splits = gkf.split(X=tdf.index, groups=tdf["business_account_number"])
        self.config["data"][prefix_out] = {}
        
        def get_chunked_dfs(dfs: dict, idx):
            """ chunk transactions df """
            dfs_ = {}
            assert("transactions" in dfs)
            for fname, df in dfs.items():
                if fname == "transactions":
                    dfs_[fname] = df.iloc[idx]
                else:
                    dfs_[fname] = df
            return dfs_ 
            
        chunk_size_tot = 0
        
        for i, (_, idx) in tqdm(enumerate(splits)):
            dfs_ = get_chunked_dfs(dfs, idx) # chunk transactions_df by index
            dfs_, self.config = joiner.run(dfs_)
            # to resolve memory issue, we store by chunks
            # make sure the returned df has the right structure
            # convert to multiple file storage structure
            assert(len(dfs_)==1 and "joined" in dfs_)  
            df = dfs_["joined"]
            chunk_size_tot += df.shape[0] # make sure combined chunk have the same size as tdf
            self.logger.info(f"chunk {i} - joined df shape: {df.shape}")
            
            dfs_ = {f"{prefix_out}_{i}": df}
            ids.extend(idx)
            
            self.save_dataframes(dfs_, self.base_path, prefix_out, self.time_id, include_subdir=True)
            
            del dfs_, df
            gc.collect()
        
        self.logger.info(f"total output chunk size: {chunk_size_tot}")
        self.logger.info(f"ALL rows are included: {len(set(ids))} vs. {len(tdf)}")
        assert(len(set(ids)) == len(tdf))
    
        self.set_config(self.config)
        

    def join_(
        self,
        prefix_in: str = "processed",
        prefix_out: str = "joined",
        debug: bool = False,
        use_dask = False,
    ):
        """ deprecated join method
        - join all data at once - which may explode memory...
        - modified so it will join by id chunks
        """
        chunk_size = self.config["chunk_size"]
        dfs = self.load_dataframes(prefix_in, self.base_path, use_dask=use_dask)
        from src.join import Joiner
        joiner = Joiner(self.config, prefix_in, prefix_out, self.logger)
        dfs, self.config = joiner.run(dfs)

        # to resolve memory issue, we store by chunks
        # make sure the returned df has the right structure
        # convert to multiple file storage structure
        assert(len(dfs)==1 and "joined" in dfs)  
        df = dfs["joined"]
        self.logger.info(f"joined df shape: {df.shape}")
        assert("business_account_number" in self.config["meta_cols"])
        
        # unevenly sliced method!
#         ids = sorted(list(df["business_account_number"].unique()))
#         dfs = {}
        
#         ids_copy = set()
#         for i in range(math.ceil(len(ids)/chunk_size)):
#             id_group = set(ids[i*chunk_size: min((i+1)*chunk_size, len(ids))])
#             dfs[f"{prefix_out}_{i}"] = df[df["business_account_number"].isin(id_group)]
#             ids_copy = ids_copy.union(id_group)
        
#         assert(len(ids_copy) == len(ids))
        
        # equally sliced method using sklearn
        from sklearn.model_selection import GroupKFold
        
        n = math.ceil(len(df) / chunk_size)
        gkf = GroupKFold(n_splits=n)
        ids = []
        dfs = {}
        splits = gkf.split(X=df.index, groups=df["business_account_number"])
        for i, (_, idx) in enumerate(splits):
            dfs[f"{prefix_out}_{i}"] = df.iloc[idx]
            ids.extend(idx)
        
        assert(len(set(ids)) == len(df))
        self.logger.info(f"ALL rows are included: {len(set(ids))} vs. {len(df)}")
        
        self.config["data"][prefix_out] = {}
        utils._save_dataframes(self.config, dfs, self.base_path, prefix_out, self.time_id,
                               include_subdir=True)
        self.set_config(self.config)

        
    def features(
        self,
        prefix_in: str = "joined",
        prefix_out: str = "features",
        debug: bool = False,
        use_dask=False
    ):
        from src.feature import FeatureEngineering
        features = FeatureEngineering(self.config, prefix_in, prefix_out, self.logger)

        # new iterative feature engineering module
        # load
        from rdsutils.datasets import DataLoader, DataDumper
        
        # get the file list..
        self.config["data"][prefix_out] = {}
        for i, (name, path) in enumerate(self.config["data"][prefix_in].items()):
            data_path = os.path.join(self.base_path, path)
            df_ = pd.read_parquet(data_path)
            self.logger.info(f"feature df_ shape: {df_.shape}")
            dfs_, self.config = features.run({f"{prefix_out}_{i}": df_})
            print(f"{prefix_out}_{i}", dfs_[f"{prefix_out}_{i}"].shape)
            self.logger.info(os.path.join(self.base_path, prefix_out))
            self.save_dataframes(dfs_, self.base_path, prefix_out, self.time_id, include_subdir=True)
            
            # TODO: make sure this works. gc is newly added.
            del dfs_, df_
            gc.collect()
            
        self.set_config(self.config)
        
        
    def labels(
        self,
        prefix_in: str = "features",
        prefix_out: str = "labeled",
        debug: bool = False,
    ):
        from rdsutils.datasets import DataLoader
        from src.utils import drop_non_ach
        
        dir_path = utils.get_data_dir(self.config, self.config["base_path"], prefix_in)
        dl = DataLoader(dir_path)
        df = dl.get_full()
        self.logger.info(f"labeled df shape: {df.shape}")
        
        df = drop_non_ach(df)
        df.reset_index(drop=True, inplace=True)
        df.columns = utils.remove_prefixs(df.columns)  # see function detail to see which prefixs are removed
        self.logger.info(f"dropped indeterminate: {df.shape}")
        gc.collect()
        dfs = {prefix_out: df}
        
        self.logger.info(f"saving data: , {self.base_path}, {prefix_out}")
        self.logger.warning("Combining the data may explode memory!")
        # this may explode memory...
        
        self.save_dataframes(
            dfs, self.base_path, prefix_out, self.time_id, include_subdir=True
        )
        
        
    def postprocess(
        self,
        prefix_in: str = "labeled",
        prefix_out: str = "postprocessed",
        debug: bool = False,
    ):
        raise NotImplemented


    def all_data_to_s3(self, base_path="data"):
        """
        put all current data object to s3
        """
        for prefix in self.config[base_path].keys():
            for name, path in tqdm(self.config[base_path][prefix].items()):
                path = os.path.join(self.base_path, path)
                s3_path = os.path.join(self.config["s3_base_path"], path)
                utils.upload_s3(
                    bucket_name=self.config["s3_bucket"],
                    path_local=path,
                    path_s3=s3_path,
                )

