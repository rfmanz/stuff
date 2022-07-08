import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
import src.utils as utils
import gc


class Joiner:
    def __init__(self, config, prefix_in, prefix_out, logger):
        self.config = config
        self.prefix_in = prefix_in
        self.prefix_out = prefix_out
        self.logger = logger
        self.dup_token = "_<DUP>"

    def run(self, dfs):
        dfs, self.config = self._run(dfs)
        return dfs, self.config

    def _run(self, dfs):
        time_id = self.config["time_id"]

        banking_df = dfs["banking"]
        ecp_df = dfs["experian_credit_pull"]
        giact_df = dfs["giact"]
        socure_df = dfs["socure"]
        transactions_df = dfs["transactions"]
        user_metadata_dw_df = dfs["user_metadata_dw"]
        bar_df = dfs["banking_account_restrictions"]
        plaid_df = dfs["plaid"]
        # quovo_df = dfs["quovo"]

        # transaction join user_meta_data
        cols = ["borrower_id", "user_id", "user_meta_sofi_employee_ind"]
        base_df = pd.merge(transactions_df,
                           user_metadata_dw_df[cols],
                           how="inner",
                           on="user_id",
                           suffixes=("", self.dup_token))

        # join banking
        self.logger.info("joining banking")
        base_df = pd.merge(base_df,
                           banking_df,
                           how="inner",
                           on="business_account_number",
                           suffixes=("", self.dup_token))
        base_df = base_df.dropna(subset=["transaction_datetime", "user_id"])

        base_df.sort_values(by=["transaction_datetime"], inplace=True)
        # join experian
        self.logger.info("joining experian")
        ecp_df = ecp_df.dropna(subset=["user_id", "ecp_credit_pull_date"])
        ecp_df.sort_values(by=["ecp_credit_pull_date"], inplace=True)
        base_df = pd.merge_asof(base_df,
                           ecp_df,
                           left_on="transaction_datetime",
                           right_on="ecp_credit_pull_date",
                           by="user_id",
                                suffixes=("", self.dup_token))

        # join giact
        self.logger.info("joining giact")
        giact_df = giact_df.dropna(subset=["giact_created_date", "user_id"])
        giact_df.sort_values(by=["giact_created_date"], inplace=True)
        base_df = pd.merge_asof(base_df,
                                giact_df,
                                left_on="transaction_datetime",
                                right_on="giact_created_date",
                                by="business_account_number",
                                suffixes=("", self.dup_token))

        # join quovo
#         self.logger.info("joining quovo")
#         quovo_df = quovo_df.dropna(subset=["quovo_current_as_of_dt", "user_id"])
#         quovo_df.sort_values(by=["quovo_current_as_of_dt"], inplace=True)
#         base_df = pd.merge_asof(base_df, quovo_df,
#                                 left_on="transaction_datetime",
#                                 right_on="quovo_current_as_of_dt",
#                                 by="user_id",
#                                 suffixes=("", self.dup_token))

        # join plaid
        self.logger.info("joining plaid")
        plaid_df = plaid_df.dropna(subset=["plaid_current_as_of_dt", "user_id"])
        plaid_df.sort_values(by=["plaid_current_as_of_dt"], inplace=True)
        base_df = pd.merge_asof(base_df, plaid_df,
                                left_on="transaction_datetime",
                                right_on="plaid_current_as_of_dt",
                                by="user_id",
                                suffixes=("", self.dup_token))

        # join socure
        self.logger.info("joining socure")
        socure_df = socure_df.dropna(subset=["socure_created_dt", "user_id"])
        socure_df.sort_values(by=["socure_created_dt"], inplace=True)
        base_df = pd.merge_asof(base_df, socure_df,
                                left_on="transaction_datetime",
                                right_on="socure_created_dt",
                                by="user_id",
                                suffixes=("", self.dup_token))

        # join tmx

        # join banking_account_restrictions_df
        self.logger.info("joining banking account restrictions")
        bar_df = bar_df.dropna(subset=["business_account_number"])
        base_df = pd.merge(base_df, bar_df,
                                how="inner", on="business_account_number", suffixes=("", self.dup_token))

        keep_cols = base_df.columns[~base_df.columns.str.contains(self.dup_token)]
        base_df = base_df[keep_cols]
        return {"joined": base_df}, self.config
