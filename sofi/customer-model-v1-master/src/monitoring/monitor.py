import sys, os, json, tqdm
import numpy as np
import pandas as pd
import datetime as dt
from collections import OrderedDict
import matplotlib.pyplot as plt

from ml4risk.monitor import MonitorBase
from ml4risk.model_selection.report import ModelReport
import ml4risk.utils as mu

import src.monitoring.refit as refit
import src.monitoring.governance as gvrn
import src.monitoring.monitoring as mntr
import src.monitoring.utils as customer_mu

plt.style.use('seaborn')


########################################
#          Report Class
########################################


class CustomerRiskV1Report(ModelReport):
    
    def __init__(self, 
                 df:pd.DataFrame,
                 target_col:str,
                 model_score_col: str,
                 baseline_cols:list,
                 artifact_path:str, 
                 model_name:str,
                 context:dict={},
                 **kwargs):
        """
        context must contains:
            target_col
            baseline_cols
            
        @params df: pd.DataFrame 
            dataframe that contains every needed info
        @params artifact_path: str
            where to store things
        @params context: dict
            parameters needed 
        """
        context["target_col"] = target_col
        context["model_score_col"] = model_score_col
        context["baseline_cols"] = baseline_cols
        context["model_name"] = model_name
        if "s3" not in artifact_path:
            os.makedirs(artifact_path, exist_ok=True)
        ModelReport.__init__(self, df, artifact_path, context, **kwargs)
        
        
    def get_pred_reports(self, context):
        """
        get prediction reports
        
        input context must contain "target_col" and "baseline_cols"
        
        @params context: dict
        @return context: dict
        """
        pred_cols = context["baseline_cols"]+[context["model_score_col"]]
        report = mntr.get_pred_reports(self.df,
                                       context["target_col"], 
                                       pred_cols, 
                                       dropna=True)
        report.to_csv(os.path.join(self.artifact_path, "performance.csv"))
        context["performance"] = report
        return context
    
    def get_segmented_perf(self, context):
        mntr.get_segmented_performances(
            {context["model_name"]: self.df},
            context["target_col"],
            [context["model_score_col"]],  # list of strs
            self.artifact_path,
            model_name=context["model_name"]
        )
        return context
        
    def get_model_vs_baseline(self, context):
        mntr.get_model_vs_baseline(
            self.df[self.df.fico_score.between(300,850)], # hard-coded...only on valid fico
            context["target_col"],
            context["model_score_col"],  # str
            "fico_adjusted",
            self.artifact_path,
        )
        return context
        
    
    def get_plots(self, context):
        # auc/pr plots - on valid fico and sigma

        plot_df = self.df.copy()
        pred_cols = context["baseline_cols"]+[context["model_score_col"]]
        
        plot_df = plot_df[
            ~plot_df[pred_cols].isna().any(axis=1) 
            & plot_df["fico_score"].between(300, 850)
        ]
        mntr.save_valid_performance_plots(
            {context["model_name"]: plot_df}, context["target_col"], 
            pred_cols, self.artifact_path, dropna=False
        )
        return context
    
    
    def run(self):
        self.context = self.get_pred_reports(self.context)
        self.context = self.get_segmented_perf(self.context)
        self.context = self.get_model_vs_baseline(self.context)
        self.context = self.get_plots(self.context)
        return self.context
    
    
########################################
#          Monitor Class
########################################

class CustomerRiskV1Monitor(MonitorBase):
    
    def __init__(self, artifact_path:str, context:dict={}, debug=False):
        MonitorBase.__init__(self, artifact_path, context)
        self.debug = debug
        
    def get_dev_data(self, context:dict):
        """
        get dev data. 
        
        @params context: dict
            must contain {"df_dev_path": path_to_data}
        """
        print("---------- getting dev data ----------")
        
        # load data queried on the dev side
        df_dev = pd.read_feather(context["df_dev_path"])
        config = context["config"]
        
        print("df_dev shape: ", df_dev.shape)
        print("sample_date max: ", df_dev.sample_date.max())
        
        # dates used to segment dev data
        mntr_start_dt = mu.monthdelta(dt.datetime.now(), -7)  # type datetime
        static_sample_dates = config["static_sample_dates"]
        date_sample_end = pd.to_datetime(config["date_sample_end"])
        context["mntr_start_dt"] = mntr_start_dt
        context["static_sample_dates"] = static_sample_dates
        context["date_sample_end"] = date_sample_end
        
        # set storage path:
        print("parsing dev dataframe")
        train_df = df_dev[df_dev.sample_date <= date_sample_end]  # training
        valid_df = df_dev[df_dev.sample_date > date_sample_end]  # after training
        mntr_df = df_dev[(df_dev.sample_date >= mntr_start_dt)]
        
        context["dev_train_df_path"] = os.path.join(self.artifact_path, "dev_train_df.parquet") 
        context["dev_valid_df_path"] = os.path.join(self.artifact_path, "dev_valid_df.parquet")
        context["dev_monitor_df_path"] = os.path.join(self.artifact_path, "dev_monitor_df.parquet")
        context["dev_full_path"] = os.path.join(self.artifact_path, "dev_df_full.parquet")
        
        print("save to s3")
        train_df.to_parquet(context["dev_train_df_path"])
        valid_df.to_parquet(context["dev_valid_df_path"])
        mntr_df.to_parquet(context["dev_monitor_df_path"])
        df_dev.to_parquet(context["dev_full_path"])
        
        return context
    
    def get_prod_data(self, context:dict):
        print("---------- getting prod data ----------")
        
        try:
            prod_df_path = os.path.join(self.artifact_path, "prod-data.parquet")
            prod_df = pd.read_parquet(prod_df_path)
        except FileNotFoundError:
            from src.monitoring.download_customer_prod import download_customer_prod_data
            download_customer_prod_data(prod_df_path)
            prod_df = pd.read_parquet(prod_df_path)
        
        context["prod_df_path"] = prod_df_path
        return context
    
    
    def get_pred(self, context:dict):
        print("---------- getting predictions using model currently in production ----------")
        
        # load monitoring snapshot dfs from s3
        mntr_df = pd.read_parquet(context["dev_monitor_df_path"])
        mntr_dfs = OrderedDict()
        
        # load model
        model_path = "jxu/money-risk-models/customer-risk-model/models/customer_risk_target_no_giact_time_since_last_link.pkl"
        model = customer_mu.read_pickle_from_s3("sofi-data-science", model_path)
        
        # score on dev data too
        df_dev = pd.read_parquet(context["dev_full_path"])
        df_dev = customer_mu.prep_customer_data(df_dev)
        df_dev["model_pred"] = model.predict_proba(df_dev[model.feature_name_])[:,1]
        df_dev["model_score"] = customer_mu.scale_scores(df_dev["model_pred"])
        context["df_dev_psi_scored_path"] = os.path.join(self.artifact_path, "df_dev_psi_scored.parquet")
        df_dev.to_parquet(context["df_dev_psi_scored_path"])
                
        # score on mntr snapshots
        for dt_str in tqdm.tqdm(context["static_sample_dates"]):
            date = pd.to_datetime(dt_str)
            if date >= context["mntr_start_dt"]:
                
                # select specific snapshot and preprocess
                df_ = mntr_df[mntr_df.is_static 
                              & (mntr_df.sample_date == date)]
                df_ = customer_mu.prep_customer_data(df_)
                
                # print some info
                imbal = 1/df_.target.value_counts(normalize=True).iloc[1]
                print(f"date: {date}, shape: {df_.shape}, imbalance: {round(imbal, 2)}")
                
                # make model pred and score
                df_["model_pred"] = model.predict_proba(df_[model.feature_name_])[:,1]
                df_["model_score"] = customer_mu.scale_scores(df_["model_pred"])
                
                # for context
                mntr_dfs[dt_str] = df_
                
        context["monitor_dfs"] = mntr_dfs
        return context
    
    def get_performance_report(self, context:dict):
        print("---------- getting performance results on the current model ----------")
        monitor_dfs = context["monitor_dfs"]
        
        for dt_str, df in tqdm.tqdm(monitor_dfs.items()):
            
            print(f"generating report for {dt_str}")
            # validation report for customer v1
            artifact_path = os.path.join(context["local_artifact_path"], dt_str)
            crr = CustomerRiskV1Report(df, "target", "model_score", 
                                       ["fico_adjusted_pred", "fraud_score_2"], 
                                       artifact_path, "customer_risk_v1", context)
            crr.run()
            
        # score psi 
        psi_df = self.get_score_psi(context)       
        context["score_psi"] = psi_df
        return context
    
    def get_score_psi(self, context):
        
        # get first validation df after model dev
        static_sample_dates = context["static_sample_dates"]
        last_date_sample_end = context["last_date_sample_end"] 
        dev_snapshot_date = min([pd.to_datetime(dttime)  # first validation date 
                                for dttime in static_sample_dates
                                if pd.to_datetime(dttime) > pd.to_datetime(last_date_sample_end)])
        
        df = pd.read_parquet(context["df_dev_psi_scored_path"])
        df_dev = df[df.sample_date == dev_snapshot_date]

        print("dev snapshot date: ", dev_snapshot_date)
        print("df_dev shape: ", df_dev.shape)
        
        # get production df
        df_prod = pd.read_parquet(context["prod_df_path"])
        print("df_prod shape: ", df_prod.shape)
    
        psi_df = mntr.get_psi(df_dev.model_score.to_frame(), 
                      df_prod.model_score.to_frame()).round(5)
        psi_df.to_csv(os.path.join(context["local_artifact_path"], 
                                    f"score-psi-dev{dev_snapshot_date}-prod.csv"))
        
        # plot score distributions to check with psi
        fig, ax = plt.subplots(1,1)
        df_prod.model_score.hist(bins=10, alpha=0.4, density=True, label="production", ax=ax)
        df_dev.model_score.hist(bins=10, alpha=0.4, density=True, label="development", ax=ax)
        plt.legend()
        plt.title("score distributions")
        plt.savefig(os.path.join(context["local_artifact_path"], 
                                 "score_distributions.png"))
        
        return psi_df
    
    def refit(self, context:dict):
        print("---------- refitting ----------")
        return context
    
    