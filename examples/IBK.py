import numpy as np
import pandas as pd

from pyutils import *

# data: https://www.kaggle.com/c/interbank20/data

path = '/home/r/Downloads/interbank20.zip'

# Load
read_data(path)
censo_test, censo_train, productos, rcc_test, rcc_train, sample_submission, se_test, se_train, sunat_test, sunat_train, y_train = read_data(path, True, "dt")

s = censo_test, censo_train, productos, rcc_test, rcc_train, sample_submission, se_test, se_train, sunat_test, sunat_train, y_train
s2 = 'censo_test, censo_train, productos, rcc_test, rcc_train, sample_submission, se_test, se_train, sunat_test, sunat_train, y_train'
all_dfs(s, s2)




#EDA
peek(y_train)
peek(productos)

y_train.key_value.nunique()
censo_train.key_value.nunique()

all_dfs??

dfs = 'censo_test, censo_train,  rcc_test, rcc_train, sample_submission, se_test, se_train, sunat_test, sunat_train, y_train'


train_dfs = list(pd.Series(s2)[pd.Series(s2).str.contains("train").values])
test_dfs= list(pd.Series(s2)[pd.Series(s2).str.contains("test").values])

# train_dfs = ','.join(train_dfs).strip(" ")
# test_dfs = ','.join(test_dfs).strip(" ")

train_dfs2 = censo_train, rcc_train, se_train, sunat_train, y_train

test_dfs2 = censo_test, rcc_test, se_test, sunat_test




dic= {}
for i, p in enumerate(train_dfs2):
    dic["{0}".format(train_dfs[i])] = p
keys = list(dic.keys())
values = list(dic.values())
for i,j in enumerate(dic):
    print(j, " ", "=", " ", values[i].key_value.nunique())

dic= {}
for i, p in enumerate(test_dfs2):
    dic["{0}".format(test_dfs[i])] = p
keys = list(dic.keys())
values = list(dic.values())
for i,j in enumerate(dic):
    print(j, " ", "=", " ", values[i].key_value.nunique())


read_data(path)

all_dfs(s, s2)

sunat_train.iloc[(sunat_train.value_counts("key_value")>1).index.values]

sunat_train[sunat_train.key_value==126876].duplicated().value_counts()
peek(sunat_train[sunat_train.key_value==126876].drop_duplicates())

# TODO: Drop duplicated, then join all the dfs, feature elimination


