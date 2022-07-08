import numpy as np
import pandas as pd
import src.process.processing as processing
import src.utils as utils
import gc

class Processor:
    def __init__(self, config, prefix_in, prefix_out, logger):
        self.config = config
        self.prefix_in = prefix_in
        self.prefix_out = prefix_out
        self.logger = logger

    def run(self):
        config = self._run()
        return config

    def _run_dask(self):
        raise NotImplemented
        return config

    def _run(self):
        """
        Insert processing functions in processor.py

        PREFIX FUNCTIONS to include with `process_`
        """
        # insert processes here
        # fn_names = utils.filter_by_prefix(dir(processing), "process_")
        config = self.config
        prefix_in = self.prefix_in
        prefix_out = self.prefix_out
        base_path = self.config["base_path"]
        time_id = self.config["time_id"]
        data_pull_date = pd.to_datetime(config["data_pull_date"])

        for name in config[base_path][prefix_in]:
            fn_name = f"process_{name}"
            # assert(fn_name in dir(processing))

            df = utils._load_dataframe(config, prefix_in, name, base_path)
            fn = getattr(processing, fn_name)

            self.logger.info(f"Processing {name} with processing.{fn_name}")
            # set kwargs
            if name == "banking":
                kwargs = {"data_pull_date": data_pull_date}
            else:
                kwargs = {}

            df = fn(df, **kwargs)

            config = utils._save_dataframes(
                config, {name: df}, base_path, prefix_out, time_id
            )

            # clean memory
            del df
            gc.collect()

        return config
