import pandas as pd
import numpy as np
from time import time
import snowflake.connector

from attrs import asdict, define, make_class, Factory
import functools
from typing import Optional, Tuple, NamedTuple, Union, Any, List, Dict, Type


@define
class dataloader:

    """Data loader"""

    # path: None
    config_file_path: str = ("config.json",)
    debug: bool = (False,)

    """@params: config...."""

    # if config_file_path:
    #   with open(config_file_path, "r") as f:
    #     CONFIG_FILE = json.load(f)

    def get_snowflake_connection(self):

        ctx = snowflake.connector.connect(
            user="<username>",
            password="<password>",
            host="localhost",
            port=1444,
            account="sdm",
            warehouse="DATA_SCIENCE",
            database="DW_PRODUCTION",
            protocol="http",
        )
        return ctx

    def run_query(self, sql):
        with self.get_snowflake_connection() as ctx:
            with ctx.cursor() as cs:
                cs.execute(sql)
                allthedata = cs.fetch_pandas_all()
                return allthedata
