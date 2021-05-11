from pyutils import *

# data: https://www.kaggle.com/c/interbank20/data
# Probabilidad de default de prestamo

path = '/home/r/Downloads/interbank20.zip'

# Load
#read_data(path)
# censo_test, censo_train, productos, rcc_test, rcc_train, sample_submission, se_test, se_train, sunat_test, sunat_train, y_train = read_data(path, True, "dt")
# Read only train:
censo_train,rcc_train,se_train,sunat_train,y_train,productos,sample_submission = read_data(path, True, 'dt',dataframes= 'censo_train,rcc_train,se_train,sunat_train,y_train,productos,sample_submission')


s = censo_train,rcc_train,se_train,sunat_train,y_train,productos,sample_submission
s2 = 'censo_train,rcc_train,se_train,sunat_train,y_train,productos,sample_submission'

all_dfs(s, s2)


#EDA
peek(y_train)
peek(productos)

# Unique key values by dataframe ?
y_train.key_value.nunique()
censo_train.key_value.nunique()

all_inital_columns_sin_productos = 'censo_test, censo_train,  rcc_test, rcc_train, sample_submission, se_test, se_train, sunat_test, sunat_train, y_train'


all_inital_columns_sin_productos = [x.strip(" ") for x in all_inital_columns_sin_productos.split(",")]

train_dfs = list(pd.Series(all_inital_columns_sin_productos)[pd.Series(all_inital_columns_sin_productos).str.contains("train").values])


test_dfs= list(pd.Series(all_inital_columns_sin_productos)[pd.Series(all_inital_columns_sin_productos).str.contains("test").values])


train_dfs2 = censo_train, rcc_train, se_train, sunat_train, y_train
test_dfs2 = censo_test, rcc_test, se_test, sunat_test

print("TRAIN: Count of Unique key_values")
dic= {}
for i, p in enumerate(train_dfs2):
    dic["{0}".format(train_dfs[i])] = p
keys = list(dic.keys())
values = list(dic.values())
for i,j in enumerate(dic):
    print(j, " ", "=", " ", values[i].key_value.nunique())
print()
print("TEST: Count of Unique key_values")
dic= {}
for i, p in enumerate(test_dfs2):
    dic["{0}".format(test_dfs[i])] = p
keys = list(dic.keys())
values = list(dic.values())
for i,j in enumerate(dic):
    print(j, " ", "=", " ", values[i].key_value.nunique())



dic= {}
for i, p in enumerate(train_dfs2):
    dic["{0}".format(train_dfs[i])] = p
keys_train = list(dic.keys())
values = list(dic.values())
d_train= []
for i,j in enumerate(dic):
    d_train.append( values[i].key_value.nunique())
dic= {}
for i, p in enumerate(test_dfs2):
    dic["{0}".format(test_dfs[i])] = p
keys_test = list(dic.keys())
values = list(dic.values())
d_test = []
for i,j in enumerate(dic):
    d_test.append(values[i].key_value.nunique())

pd.concat([pd.Series(keys_train),pd.Series(d_train),pd.Series(keys_test),pd.Series(d_test)],axis=1,keys=["TRAIN", "Unique","TEST","Unique"])

# We're modelling based 358497 unique clients people. These people all should have loans on the rcc

all_dfs(all_inital_columns,all_inital_columns_quoted)
describe_df(rcc_train,'.0f')

rcc_train.columns = map(str.lower, rcc_train.columns)

rcc_train.value_counts("producto")
productos
y_train.value_counts("target")

rcc_train[rcc_train.key_value.nunique()]

# The test datasets are literally just for the scoring with is to be uploaded.

peek(rcc_train)
for i in rcc_train.key_value:
    print(i)




read_data(path)

all_dfs(s, s2)

sunat_train.iloc[(sunat_train.value_counts("key_value")>1).index.values]

sunat_train[sunat_train.key_value==126876].duplicated().value_counts()
peek(sunat_train[sunat_train.key_value==126876].drop_duplicates())

# TODO: Drop duplicated, then join all the dfs, feature elimination


