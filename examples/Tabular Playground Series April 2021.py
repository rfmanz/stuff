from pyutils import *
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import *
# from sklearn.metrics import *
# from sklearn.model_selection import *
# import lightgbm as lgb

read_data('/home/r/Downloads/tabular-playground-series-apr-2021.zip')
sample_submission,test,train = read_data('/home/r/Downloads/tabular-playground-series-apr-2021.zip',True)
sample_submission.shape,test.shape,train.shape

train.hist()

#sibsp = # of siblings / spouses aboard the Titanic
#parch =  # of parents / children aboard the Titanic
