# region /// THIS IS THE WAY ///
# data: https://www.kaggle.com/c/interbank20/data
# Probabilidad de default de prestamo
import matplotlib.pyplot as plt

from pyutils import *

pd_options()

path = 'D:/Downloads/interbank20.zip'
y_train = read_data(path, True,'dt',dataframes="y_train")
y_train = y_train.target*1

# region /// rcc_train ///

rcc_test,rcc_train = read_data(path, True, 'dt', dataframes="rcc_test,rcc_train")
#rcc_train = read_data(path, True, 'dt', dataframes="rcc_train")

reduce_memory_usage(rcc_train)
reduce_memory_usage(rcc_test)
# View
rcc_train.head()
peek(rcc_train)
# First thing we want to do is actually check if the columns are in the right dtypes. How do we know that the variables were read in with suitable data dtypes. Well by th number of unique instance that variable has. So any column which has been read in as a numerical variable and has than less than say 50 categories should really be transformed into a categorical value.
rcc_train.nunique()
dtypes(rcc_train)

def convert_dtypes(df,varsN , varsC ):

    df[varsN] = df[varsN].astype(int)
    df[varsC] = df[varsC].astype('category')
    #df[varsS] = df[varsS].astype(str)

    return df

varsN = ['condicion','saldo']
varsC = ['tipo_credito','cod_instit_financiera','PRODUCTO','RIESGO_DIRECTO','COD_CLASIFICACION_DEUDOR',
         'key_value']

rcc_train = convert_dtypes(rcc_train,varsN,varsC)
dtypes(rcc_train)


# Fixing stuff
rcc_train.columns = rcc_train.columns.str.lower()
productos = read_data(path,True, 'dt', dataframes="productos")
# column added for visualization
rcc_train = rcc_train.merge(
    productos, how='left', left_on="producto", right_on="Productos")
rcc_train.rename(columns={"C0": "productos_nm"}, inplace=True)
rcc_train.drop(columns="Productos", inplace=True)
rcc_train.productos_nm.fillna("NULL", inplace=True)
rcc_train.drop(rcc_train[rcc_train.producto.astype(int).isin([36, 41])].index, inplace=True)
rcc_train.codmes = rcc_train.codmes.astype(str)
rcc_train = pd.merge(rcc_train, y_train, right_index=True, left_on='key_value' )


# newregion ///EDA///
# So I was initially more clever about this than I thought. Most of the graphing functions in EDA are actually ready to go for any dataset/columns which have a binary classification target.

# Why did I do make it so? If I'm going to work in data science, most things are going to be a classification problem: will the client buy/ or try to predict this outcome.
class_hists(rcc_train,"saldo","target")
plt.show()
plt.figure()
plot_single_numerical(rcc_train['saldo'])
class_hists(rcc_train,'saldo','target')
plt.show()
plot_univariate_classification(rcc_train[['target','condicion']],'target')
box_plot_classification(rcc_train[['target','saldo']],'target')

peek(rcc_train)
rcc_train.key_value.nunique()


















plt.figure(figsize=(10, 10))
sns.boxplot(rcc_train.condicion)
ax = sns.boxplot(x="", y="value", hue=target_name, data=data)

pd.cut(rcc_train.condicion, bins=5, right=True).value_counts(normalize=True).sort_index() * 100

bins = [-1, 0, 10, 30, 180, 720, float("inf")]
rcc_train["condicion_cat"] = (pd.cut(rcc_train.condicion, bins).cat.codes).astype('category')

plt.hist(rcc_train['condicion_cat'],bins = 6)
plt.show()
dtypes(rcc_train)

plt.style.use('fivethirtyeight')
plt.hist(rcc_train.sample(int(len(rcc_train)/10))['condicion'],bins=100)
rcc_train.condicion.value_counts()
rcc_train.condicion.value_counts(normalize=True)*100

plt.show()
describe_df(rcc_train[['condicion']])

ordered_barplot(rcc_train,'condicion_cat')
plot_single_numerical(rcc_train.sample(int(len(rcc_train)/100))['condicion'])



describe_df(rcc_train[['condicion_cat']])
rcc_train[['condicion_cat']].value_counts(sort=False)
rcc_train[['condicion_cat']].value_counts()

rcc_test["condicion_cat"] = pd.cut(rcc_test.condicion, bins).cat.codes

# endregion

# rcc_train = rcc_test
def diferent_vals_cat(train,test,varC):
    diferentes = {}
    uniques_train =  sorted(train[varC].unique())
    uniques_test =  sorted(test[varC].unique())
    diferentes['train'] = list(j for j in uniques_train if j not in uniques_test)
    diferentes['test'] = list(j for j in uniques_test if j not in uniques_train)
    print("*"*10, varC, "*"*10)
    print(f"Not in test: {diferentes['train']}\nNot in train: {diferentes['test']}")

rcc_train.head()
uniques_train =  sorted(rcc_train["tipo_credito"].unique())
uniques_test =  sorted(rcc_test["tipo_credito"].unique())

list(j for j in uniques_train if j not in uniques_test)
[j for j in uniques_test if j not in uniques_train]

uniques_train =  sorted(rcc_train["tipo_credito"].unique())

list(set(rcc_train.tipo_credito) - set(rcc_test.tipo_credito))
list(set(rcc_train.tipo_credito) - set(rcc_test.tipo_credito))

peek(rcc_train)
peek(rcc_test)
rcc_train.dtypes

rcc_train.codmes =  rcc_train.codmes.astype(str)
rcc_train.dtypes

rcc_train.tipo_credito.value_counts()

rcc_train.nunique()
describe_df(rcc_train)
diferent_vals_cat(rcc_train,rcc_test,'tipo_credito')
diferent_vals_cat(rcc_train,rcc_test,'key_value')


rcc_train.dtypes
rcc_test.dtypes

rcc_train.drop_duplicates(inplace=True)
rcc_test.drop_duplicates(inplace=True)
sunat_train.drop_duplicates(inplace=True)
sunat_test.drop_duplicates(inplace=True)
censo_train.drop_duplicates(inplace=True)
censo_test.drop_duplicates(inplace=True)

rcc_train, productos, rcc_train = rcc_train()
rcc_train.head()

rcc_train[["producto", 'productos_nm']].value_counts().sort_values()
rcc_train.groupby(["productos_nm"]).mean("saldo").sort_values("saldo")
rcc_train.productos_nm
rcc_train.loc[(rcc_train.PRODUCTO.astype(int).isin([36,41])),'key_value'].nunique()
rcc_train.loc[(rcc_train.PRODUCTO.astype(int).isin([36,41]))]

rcc_train.cod_clasificacion_deudor.astype(object)
rcc_train.riesgo_directo.value_counts()
encode(pd.DataFrame(rcc_train[["riesgo_directo"]]))

#endregion


#region /// socio_economico ///

se_train = read_data(path, True, 'dt', 'se_train')
describe_df(se_train)
se_train.astype("object").describe()
describe_df(se_train)
#endregion



#-----
rcc_train.columns
peek(rcc_train2)
y_train.dtypes
wwtrain =  rcc_train.merge(censo_train, on="key_value", how='left')

rcc_train.merge(se_train, on='key_value', how='left').merge(sunat_train, on = 'key_valye', how= 'left')

all_dfs(s, s2)

all_dfs(s, s2)

train = pd.merge([rcc_train, se_train, sunat_train, censo_train], on='key_value', )
# test = pd.concat([censo_tes, rcc_train, se_train, sunat_train], axis=0)
y = y_train

set(y_train.index) in set(rcc_train.key_value)

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
    dic["{0}".format(train_dfs[i])] =
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
