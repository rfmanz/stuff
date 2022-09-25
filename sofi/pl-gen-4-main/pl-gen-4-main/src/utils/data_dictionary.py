# auther: George Xu
# https://gitlab.com/sofiinc/data-science-risk/poc-tools/ml4risk/-/blob/master/ml4risk/data_preparation/data_dictionary.py
# update (Hua): add process_attr_group
import sys
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod


class DataDictionaryBase(metaclass=ABCMeta):
    def __init__(
        self,
        df,
        name_col,
        type_col,
        description_col,
        valid_val_col,
        default_exclusions_col,
    ):
        self.dd = df
        self.name_col = name_col
        self.type_col = type_col
        self.description_col = description_col
        self.default_exclusions_col = default_exclusions_col
        self.valid_val_col = valid_val_col

    @abstractmethod
    def process_default_exclusions(self, series):
        pass

    @abstractmethod
    def process_numerical(self, series):
        pass

    @abstractmethod
    def process_data_type(self, series):
        pass


class ExperianDataDict(DataDictionaryBase):
    def __init__(
        self,
        df,
        name_col,
        type_col,
        description_col,
        valid_val_col,
        default_exclusions_col,
    ):
        super().__init__(
            df,
            name_col=name_col,
            type_col=type_col,
            description_col=description_col,
            default_exclusions_col=default_exclusions_col,
            valid_val_col=valid_val_col,
        )
        self.dd = self.process_dd(df)

    def process_default_exclusions(self, series, dtype):
        def cat_parser(x):
            x = str(x).replace("[", "").replace("]", "").replace("'", "")
            L = x.split("\n")
            L = [l.strip().split(":")[0] for l in L]
            L = [int(l) for l in L if l.isnumeric()]
            if len(L) == 0:
                return None
            return L

        return series.apply(cat_parser)

    def process_numerical(self, series, dtype):
        df_ = series.to_frame()
        df_.columns = ["valid_val"]
        df_["type"] = dtype

        def get_min(row):
            try:
                if str(row["valid_val"]).lower() in ["nan", "none"]:
                    return np.nan
                if (row["type"] == "int") and isinstance(row["valid_val"], str):
                    min_max = row["valid_val"].split("-")
                    if len(min_max) != 2:
                        return "MANUAL TREATMENT NEEDED"
                    return int(min_max[0].strip())
            except:
                return "MANUAL TREATMENT NEEDED"
            return "MANUAL TREATMENT NEEDED"

        def get_max(row):
            try:
                if str(row["valid_val"]).lower() in ["nan", "none"]:
                    return np.nan
                if (row["type"] == "int") and isinstance(row["valid_val"], str):
                    min_max = row["valid_val"].split("-")
                    if len(min_max) != 2:
                        return "MANUAL TREATMENT NEEDED"
                    return int(min_max[1].strip())
            except:
                return "MANUAL TREATMENT NEEDED"
            return "MANUAL TREATMENT NEEDED"

        df_["min"] = df_.apply(get_min, axis=1)
        df_["max"] = df_.apply(get_max, axis=1)
        return df_["min"], df_["max"]
    
#    def process_attr_group(self):
#        df_ = self.dd.copy()
#        df_['attr_grp'] = None
#        df_.loc[df_[self.description_col].str.contains("""delinquent|delinquency|derogatory|repossession|charge-off
#                                                 |Worst|worst|charged-off|foreclosure|repossessed
#                                                 |30|60|90|120|150|180
#                                                 |satisfactory|unsatisfied|past due
#                                                 """,na=False),'attr_grp'] = 'payment history' 
#        df_.loc[df_[self.description_col].str.contains("""opened in|since the most recent|since the oldest|number of months
#                                                """,na=False),'attr_grp'] = 'credit history'
#        df_.loc[df_[self.description_col].str.contains('inquir',na=False),'attr_grp'] = 'inquiry'
#        df_.loc[df_[self.description_col].str.contains('ratio |Ratio |balance to|Balance to',na=False)
#                ,'attr_grp'] = 'credit utilization'
#        df_.loc[df_[self.name_col].str.contains('trended3d',na=False),'attr_grp'] = 'trended'
#        df_.loc[df_.attr_grp.isnull(),'attr_grp'] = 'credit mix'
#        return df_['attr_grp']
    
    def process_premier_attr_group(self):
        df_=self.dd.copy()
        df_['premier_attr_num'] = np.where(df_['table_name']=='premier_1_3',df_['field_name'].str[7:8],np.nan)
        mapper={
                    "0": "Trade Count",
                    "1": "Satisfactory Trade Count",
                    "2": "Delinquent Trade Count",
                    "3": "Other Criteria Counts",
                    "4": "Occurrence",
                    "5": "Balance Amount Payment",
                    "6": "Rank",
                    "7": "Ratios/Percentage",
                    "8": "Age/Recency",
                    "9": "Public Records/Inquiry"
                }
        df_['attr_grp'] = df_['premier_attr_num'].map(mapper)        
        return df_['attr_grp']
    
    def process_trend11_attr_group(self):
        df_=self.dd.copy()
        df_['attr_grp'] = np.where(df_['table_name']=='trended_3d_v1_1',df_['concept'],np.nan)      
    
        return df_['attr_grp']
    
    def process_trend_attr_group(self):
        df_=self.dd.copy()
        df_['attr_grp'] = np.where(df_['table_name']=='trended_3d','trended_3d',np.nan)       
        return df_['attr_grp']
    
    def process_data_type(self, series):
        from pandas._libs.tslibs.timestamps import Timestamp

        types = series.astype(str)
        mapper = {"int": "int", "varchar": "str", "bigint": "int", "date": "date"}
        types.replace(mapper, inplace=True)
        return types
    
    def process_attr_group(self):
        df_=self.dd.copy()
        df_['pre_attr_grp'] = self.process_premier_attr_group()
        df_['trend11_attr_grp'] = self.process_trend11_attr_group()
        df_['trend_attr_grp'] = self.process_trend_attr_group()
        df_['attr_grp'] = np.where(df_['table_name']=='premier_1_3', df_['pre_attr_grp'],
                                  np.where(df_['table_name']=='trended_3d_v1_1', df_['trend11_attr_grp'],df_['trend_attr_grp']))
        return df_['attr_grp']
        
    def process_dd(self, df):
        dd = df.copy()
        dd["type"] = self.process_data_type(dd[self.type_col])
        dd["categorical"] = self.process_default_exclusions(
            dd[self.default_exclusions_col], dd["type"]
        )
        dd["min"], dd["max"] = self.process_numerical(
            dd[self.valid_val_col], dd["type"]
        )
        dd["attr_grp"] = self.process_attr_group()
        return dd

    @property
    def data_dict(self):
        return self.dd