from pyutils import *

# data: https://www.kaggle.com/c/interbank20/data
# Probabilidad de default de prestamo

path = '/home/r/Downloads/interbank20.zip'

# Load
# read_data(path)
# censo_test, censo_train, productos, rcc_test, rcc_train, sample_submission, se_test, se_train, sunat_test, sunat_train, y_train = read_data(path, True, "dt")
# Read only train:

# Only load train
censo_train, rcc_train, se_train, sunat_train, y_train, productos, sample_submission = read_data(path, True, 'dt',
                                                                                                 dataframes='censo_train,rcc_train,se_train,sunat_train,y_train,productos,sample_submission')

s = censo_train, rcc_train, se_train, sunat_train
s2 = 'censo_train,rcc_train,se_train,sunat_train'
all_dfs(s, s2)

# rcc_train
peek(rcc_train)
rcc_train.columns = map(str.lower, rcc_train.columns)
for i in [censo_train, rcc_train, se_train, sunat_train]:
    print(peek(i))

d = [censo_train, rcc_train, se_train, sunat_train]
from functools import reduce

df_merged = reduce(lambda left, right: pd.merge(left, right, on=['key_value']), d)


def memory_usage_mb(df, *args, **kwargs):
    """Dataframe memory usage in MB. """
    return df.memory_usage(*args, **kwargs).sum()


def reduce_memory_usage(df, deep=True, verbose=True, categories=True):
    # All types that we want to change for "lighter" ones.
    # int8 and float16 are not include because we cannot reduce
    # those data types.
    # float32 is not include because float16 has too low precision.
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = memory_usage_mb(df, deep=deep)

    for col, col_type in df.dtypes.iteritems():
        best_type = None
        if col_type == "object":
            df[col] = df[col].astype("category")
            best_type = "category"
        elif col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name
        # Log the conversion performed.
        # if verbose and best_type is not None and best_type != str(col_type):
        # print(f"Column {col} converted from {col_type} to {best_type}")

    if verbose:
        end_mem = memory_usage_mb(df, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(f"Memory usage decreased from"
              f" {start_mem:.2f}MB to {end_mem:.2f}MB"
              f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")


for i in d:
    reduce_memory_usage(i)

print(f"{(sys.getsizeof(rcc_train) / 1024 ** 2):.2f} Mb")

reduce_memory_usage(rcc_train)

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
