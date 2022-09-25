import os, sys
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod


class DataloaderBase(metaclass=ABCMeta):
    def __init__(self, train_path=None, valid_path=None, test_path=None, **kwargs):
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """load data from paths provided"""
        raise NotImplementedError

    @abstractmethod
    def process(self, *args, **kwargs):
        """process loaded data

        - debug mode
        - batchify
        - transformations if needed
        """
        raise NotImplementedError


class TabularDataloader(DataloaderBase):
    def __init__(
        self, train_path, valid_path=None, test_path=None, target_col=None, **kwargs
    ):
        super().__init__(train_path, valid_path, test_path, **kwargs)
        self.target_col = target_col
        self.debug_size = None

        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.debug_train_df = None
        self.debug_valid_df = None
        self.debug_test_df = None
        for k in kwargs:
            setattr(k, kwargs[k])

    def load_data(self, debug_size=None, random_state=42):
        self.debug_size = debug_size

        self.train_df = self.load_df(self.train_path)
        if self.valid_path:
            self.valid_df = self.load_df(self.valid_path)
        if self.test_path:
            self.test_df = self.load_df(self.test_path)

        if debug_size is not None and debug_size < 1:
            self.debug_train_df = self.train_df.sample(
                frac=debug_size, random_state=random_state
            )
            if self.valid_df is not None:
                self.debug_valid_df = self.valid_df.sample(
                    frac=debug_size, random_state=random_state
                )
            if self.test_df is not None:
                self.debug_test_df = self.test_df.sample(
                    frac=debug_size, random_state=random_state
                )
        elif debug_size is not None and debug_size >= 1:
            self.debug_train_df = self.train_df.sample(
                n=debug_size, random_state=random_state
            )
            if self.valid_df is not None:
                self.debug_valid_df = self.valid_df.sample(
                    n=debug_size, random_state=random_state
                )
            if self.test_df is not None:
                self.debug_test_df = self.test_df.sample(
                    n=debug_size, random_state=random_state
                )

    def process(self):
        pass

    def get_data(self, debug=False):
        if debug:
            return (self.debug_train_df, self.debug_valid_df, self.debug_test_df)
        else:
            return (self.train_df, self.valid_df, self.test_df)

    def get_shapes(self):
        msgm = []
        for df in [self.train_df, self.valid_df, self.test_df]:
            if df is not None:
                msgm.append(df.shape)
        return str(msgm)

    @staticmethod
    def load_df(path, **kwargs):
        if str(path).endswith(".parquet"):
            return pd.read_parquet(path, **kwargs)
        elif str(path).endswith(".feather"):
            return pd.read_parquet(path, **kwargs)
        elif str(path).endswith(".csv"):
            return pd.read_csv(path, **kwargs)
        else:
            ValueError("Unknown input types")
