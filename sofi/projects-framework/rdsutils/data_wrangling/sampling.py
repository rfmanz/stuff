import pandas as pd
import numpy as np


def get_spacing(dates):
    """ Use this to find the spacing of previous sampling methods """
    dates = pd.to_datetime(dates)
    dates = pd.Series(dates).diff().value_counts()
    return dates

def get_sampling_dates(start, end, freq):
    """ Get static sampling dates from start to end with period in between """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end).normalize()
    
    result = list(pd.date_range(start, end, freq=freq))
    result = list(map(lambda d: str(d).split(" ")[0], result))
    return result

def get_monitoring_dates(start, end="today"):
    """ We get monitoring dfs by looking at first day of every month """ 
    start = pd.to_datetime(start)
    end = pd.to_datetime(end).normalize()
    
    # 365/28 about 13, so set 15 to include every month
    dates = pd.date_range(start, end, freq="15D")
    dates = sorted(list(set(map(lambda d: d.replace(day=1), dates))))
    dates = list(map(lambda d: str(d).split(" ")[0], dates))
    
    return dates
