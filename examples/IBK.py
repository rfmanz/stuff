from pyutils import *

# data: https://www.kaggle.com/c/interbank20/data
# Probabilidad de default de prestamo

path = 'D:/Downloads/interbank20.zip'

# Load
read_data(path)
# censo_test, censo_train, productos, rcc_test, rcc_train, sample_submission, se_test, se_train, sunat_test, sunat_train, y_train = read_data(path, True, "dt")
# Read only train:

# Only load train
# censo_train, rcc_train, se_train, sunat_train, y_train, productos, sample_submission = read_data(path, True, 'dt', dataframes='censo_train,rcc_train,se_train,sunat_train,y_train,productos,sample_submission')


# rcc_train
rcc_train,y_train,productos = read_data(path, True,'dt',dataframes="rcc_train,y_train,productos")
#TODO: Fix read_data function. When only one dataframe name is passed it returns dictionary insteado of dataframe.
y_train = y_train.target*1


rcc_train2 = rcc_train.copy()
reduce_memory_usage(rcc_train2)
rcc_train2.columns = rcc_train2.columns.str.lower()

np.setdiff1d(productos.Productos,rcc_train2.producto.unique())
np.setdiff1d(rcc_train2.producto.unique(),productos.Productos)

rcc_train2 = rcc_train2.merge(productos, how = 'left',left_on="producto",right_on="Productos")

rcc_train2.rename(columns= {"C0":"productos_nm"},inplace=True)
rcc_train2.drop(columns="Productos",inplace=True)

rcc_train2.groupby('productos_nm')['saldo'].mean().plot.bar()
productos.C0.str.contains("FORWARDS")
rcc_train2.loc[np.where(rcc_train2.productos_nm.str.contains("FORWARDS"))]
rcc_train2.loc[rcc_train2.productos_nm.dropna().str.contains("FORWARDS")]

rcc_train2.productos_nm.filter(like='FORWARDS',axis=0)
rcc_train2.productos_nm.str.contains("FORWARDS").value_counts()
rcc_train2.isnull().sum()

rcc_train2[rcc_train2['productos_nm'].str.contains("FORWARDS",na=False)]
rcc_train2.loc[rcc_train2['productos_nm'].str.contains("FORWARDS",na=False),['saldo']].mean()

rcc_train2.groupby("productos_nm")['saldo'].mean().sort_values()

rcc_train2.key_value.nunique()

rcc_train2.productos_nm.value_counts(dropna=False)

rcc_train2.productos_nm.value_counts()

productos.C0
rcc_train.producto.nunique()
productos.C0.nunique()



rcc_train.columns
peek(rcc_train2)
y_train.dtypes
train =  rcc_train.merge(censo_train, on="key_value", how='left')

    .merge(se_train, on='key_value', how='left').merge(sunat_train, on = 'key_valye', how= 'left')

all_dfs(s, s2)

all_dfs(s, s2)

train = pd.merge([rcc_train, se_train, sunat_train, censo_train], on='key_value', )
# test = pd.concat([censo_tes, rcc_train, se_train, sunat_train], axis=0)
y = y_train

set(y_train.key_value) in set(rcc_train.key_value)

set(rcc_train.key_value.unique()) in set(y_train.key_value)

rcc_train[rcc_train.key_value.unique()]

list(set(rcc_train.key_value.unique()).difference(y_train.key_value))

list(set(y_train.key_value).difference(sunat_train.key_value))

len(list(set(y_train.key_value).difference(censo_train.key_value.unique())))

censo_train.key_value.nunique()

list(set(se_train.key_value).difference(y_train.key_value))

list(set(y_train.key_value).difference(rcc_train.key_value.unique()))

# censo train nos falta data de 205348 personas. Vale la pena incluir esta tabla en el analisis ?


set(rcc_train.key_value.unique()) in set(y_train.key_value)

pd.Series.rcc_train.key_value.unique().str.match() in set(y_train.key_value)

# EDA
describe_df(rcc_train)
productos_dic = productos.to_dict()
productos_dic.keys()
rcc_train.prod

peek(y_train)
peek(productos)

# Unique key values by dataframe ?
all_inital_columns_sin_productos = 'censo_test, censo_train,  rcc_test, rcc_train, sample_submission, se_test, se_train, sunat_test, sunat_train, y_train'

all_inital_columns_sin_productos = [x.strip(" ") for x in all_inital_columns_sin_productos.split(",")]

train_dfs = list(pd.Series(all_inital_columns_sin_productos)[
                     pd.Series(all_inital_columns_sin_productos).str.contains("train").values])

test_dfs = list(pd.Series(all_inital_columns_sin_productos)[
                    pd.Series(all_inital_columns_sin_productos).str.contains("test").values])

train_dfs2 = censo_train, rcc_train, se_train, sunat_train, y_train
test_dfs2 = censo_test, rcc_test, se_test, sunat_test

print("TRAIN: Count of Unique key_values")
dic = {}
for i, p in enumerate(train_dfs2):
    dic["{0}".format(train_dfs[i])] = p
keys = list(dic.keys())
values = list(dic.values())
for i, j in enumerate(dic):
    print(j, " ", "=", " ", values[i].key_value.nunique())
print()
print("TEST: Count of Unique key_values")
dic = {}
for i, p in enumerate(test_dfs2):
    dic["{0}".format(test_dfs[i])] = p
keys = list(dic.keys())
values = list(dic.values())
for i, j in enumerate(dic):
    print(j, " ", "=", " ", values[i].key_value.nunique())

dic = {}
for i, p in enumerate(train_dfs2):
    dic["{0}".format(train_dfs[i])] = p
keys_train = list(dic.keys())
values = list(dic.values())
d_train = []
for i, j in enumerate(dic):
    d_train.append(values[i].key_value.nunique())
dic = {}
for i, p in enumerate(test_dfs2):
    dic["{0}".format(test_dfs[i])] = p
keys_test = list(dic.keys())
values = list(dic.values())
d_test = []
for i, j in enumerate(dic):
    d_test.append(values[i].key_value.nunique())

pd.concat([pd.Series(keys_train), pd.Series(d_train), pd.Series(keys_test), pd.Series(d_test)], axis=1,
          keys=["TRAIN", "Unique", "TEST", "Unique"])

# We're modelling based 358497 unique clients people. These people all should have loans on the rcc

all_dfs(all_inital_columns, all_inital_columns_quoted)
describe_df(rcc_train, '.0f')

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

sunat_train.iloc[(sunat_train.value_counts("key_value") > 1).index.values]

sunat_train[sunat_train.key_value == 126876].duplicated().value_counts()
peek(sunat_train[sunat_train.key_value == 126876].drop_duplicates())

# TODO: Drop duplicated, then join all the dfs, feature elimination
