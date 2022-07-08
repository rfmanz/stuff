import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count


def applyParallel(dfGrouped, func):
    """ Helper to parallelize apply over groupby """
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)


def get_full_column(df, s):
    """ to get the new column name with prefix attached... """
    return df.columns[df.columns.str.contains(s)]