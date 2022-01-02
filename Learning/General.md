# Data Exploration



## Check dataframe

```python
data.head```

## Dataframe Shape
```python
data.shape```

## Dataframe Columns

```
data.columns
list(data.columns.values)```






 ## Filter or Select Column

```data['education'].unique()```

And = &    Or= | 
![image.png](attachment:image.png)




Same

![image.png](attachment:image.png)





Select column 

```python

r.filter(regex="DEPARTAMENTO_CONTRA")

r2.filter(["DEPARTAMENTO_CONTRA","PROVINCIA_CONTRA","DISTRITO_CONTRA"])

pol_base[(pol_base.DESCPROD.isin(["ACCIDENTES DEVOLUCIÓN PLUS BCP", "VIDA DEVOLUCIÓN PLUS BCP"]))].filter(["DESCPROD"]).shape



.loc is for selecting things by label



# dataframe.loc[whatrowsdoiwant,whatcolumns]



dft["tasa_final"][dft["tasa_final"] <= percentiles[0]] = percentiles[0] 

dft.loc[dft.index[np.where(dft["tasa_final"]<=percentiles[0])],"tasa_final"]

dft.loc[dft["tasa_final"]<=percentiles[0],"tasa_final"]



dft.columns.isin(["lgr"]).any()



df_aux.select_dtypes(include=["number"])



df.loc[:, df.columns.str.startswith('foo')]



rcc_test[(rcc_test.PRODUCTO.astype(int).isin([36,41]))]

rcc_train2[rcc_train2['productos_nm'].str.contains("FORWARDS",na=False)]



```

## Value Counts

```python
df['y'].value_counts()
asesor['NOMBRECANAL'].value_counts() 
df.key_value.astype("str").duplicated().value_counts()
```
This is the same as R table


## Dataframe in New Window


<span class="mark">data</span> is your dataframe

```python
from IPython.display import display, HTML
s  = '<script type="text/Javascript">'
s += 'var win = window.open("", "Title", "toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, width=780, height=200, top="+(screen.height-400)+", left="+(screen.width-840));'
s += 'win.document.body.innerHTML = \'' + data.to_html().replace("\n",'\\') + '\';'
s += '</script>'
HTML(s) ```

or better version perhaps 

```python
from IPython.display import HTML
def View(df):
    css = """<style>
    table { border-collapse: collapse; border: 3px solid #eee; }
    table tr th:first-child { background-color: #eeeeee; color: #333; font-weight: bold }
    table thead th { background-color: #eee; color: #000; }
    tr, th, td { border: 1px solid #ccc; border-width: 1px 0 0 1px; border-collapse: collapse;
    padding: 3px; font-family: monospace; font-size: 10px }</style>
    """
    s  = '<script type="text/Javascript">'
    s += 'var win = window.open("", "Title", "toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, width=780, height=200, top="+(screen.height-400)+", left="+(screen.width-840));'
    s += 'win.document.body.innerHTML = \'' + (data.to_html() + css).replace("\n",'\\') + '\';'
    s += '</script>'
    return(HTML(s+css))

View(data)```
    


## Group by 

```python 
data.groupby(data['y']).mean()
```

.sum


## Describe data

```python
data.describe()```

## Copy to Excel

```python
data.to_clipboard(excel=True)```

## Countif in column 

```python
data['y'][data['y']==0].count()```
also 
```python
len(data[data['y']==0])```

## Add Column names to data without column names
First
```python
df=pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header=None)```
then
```python
df.columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','hd']```



## Check Null Values

```python 
df1.isnull()
```
Count
```python
unexpected.isnull().sum()
```

By column
```python
df['ST_NUM'].isnull()
```
For all columns

```python 
null_columns=unexpected.columns[unexpected.isnull().any()]
unexpected[null_columns].isnull().sum()

def num_missing(x):
  return sum(x.isnull())
print (train.apply(num_missing, axis=0))

```

Sum all Nulls 
```python 
df.isnull().sum().sum()
```

## Memory size


```python
print(f"{(sys.getsizeof(df)/1024**2):.2f} Mb")
```





## Drop rows with NULLs

```python
df=df.dropna()
```

## Replace Values of a Column
```python
df[['sex']] = df[['sex']].replace(['M'], [1]) 
```

```python
df['sex'][df['sex']==0]='F'
```

## Make weird characters into NULL 

```python
df=df.replace("?",np.nan)
```

## Find all Rows with Nans
```python
df[df.isnull().any(axis=1)]
```
### Columns with Nans
```python 
df[df.isnull().any(axis=1)][null_columns]
```
### Specific Column
```python
train[train["Electrical"].isnull()][null_columns])
```

## Count whatever within whole dataframe
```python
df.iloc[:][df[:]=="?"].count()
```
### Value Counts
```python 
df.iloc[:,1].value_counts()
```
Value counts over the whole dataframe
```python
flg_dft.apply(pd.Series.value_counts)
flg_dft.apply(pd.Series.value_counts).transpose()

#More summarised 
train.ts_id.unique().value_counts().value_counts()
train.date.unique().value_counts().value_counts()



```


## Lambda

```python 
df['hd'] = df['hd'].apply(lambda x: 'Healhty' if x == 0 else 'Unhealthy')
```

Here we are just saying if 0 then Healthy if not Unhealthy.

## Change to Float

```python
df['ca'] = df.ca.astype(float)

s2 = s2.apply(pd.to_numeric)
```

## Groupby and Crosstab

```python
df.groupby(['hd','sex']).size().to_frame('Frequency')

pd.crosstab(df.hd, df.sex)

```


## Qgrid

```python
import qgrid
qgrid_widget=qgrid.show_grid(df, show_toolbar=True)
qgrid_widget
```

## Jupyter Theme

jt -t monokai -cellw 97% -f roboto -fs 11 -ofs 11 -T

## Logistic Regression
http://www.science.smith.edu/~jcrouser/SDS293/labs/lab4-py.html

```python
model = smf.glm(formula = formula, data=df, family=sm.families.Binomial())
result = model.fit()
print(result.summary())

import statsmodels.api as sm
Formula = 'hd ~ sex'
res=sm.formula.glm(formula=Formula,data=df, family=sm.families.Binomial()).fit()
res.summary()
```

## Unique

```python 
pd.unique(df['Country'])
```
```python 
pd.unique(df.Country)
```
```python 
df.Country.unique()
```
```python 
df.Country.nunique()
```
### Unique for every column
```python 
def unique_counts(df1):
   for i in df1.columns:
       count = df1[i].nunique()
       print(i, ": ", count)
unique_counts(df1)def unique_counts(df1):
   for i in df1.columns:
       count = df1[i].nunique()
       print(i, ": ", count)
unique_counts(df1)
```



## Find Duplicates & Drop Duplicates

```python
df[df['CustomerID'].duplicated()]
```

```python
customer_country=df[['Country','CustomerID']].drop_duplicates()
```
```python
df.loc[df.CustomerID.duplicated(),:]
```

Count duplicates
```python
df.CustomerID.duplicated().shape
```

Count duplicates by columns
```python
df.duplicated(subset=['age','w']).sum
```

## Date

```python
raw_data['Mycol'] =  pd.to_datetime(raw_data['Mycol'], format='%d%b%Y:%H:%M:%S.%f')
```

## Create New Dataframe using Columns from another Dataframe

```python
s3 = s1[['Playlist', 'Track Name']].copy()
```

## IDXMIN & IDXMAX

```python
r.Base_sin_ape.loc[r.Base_sin_ape["MONTO_PAGO_NETO_USD"].idxmin()]
r.Base_sin_ape.loc[r.Base_sin_ape["MONTO_PAGO_NETO_USD"].idxmin()]["MONTO_PAGO_NETO_USD"]
```

## ILOC
```python
>>> df.iloc[[0, 2], [1, 3]]
      b     d
0     2     4
2  2000  4000

>>> df.iloc[1:3, 0:3]
      a     b     c
1   100   200   300
2  1000  2000  3000


```

```python

prices = pd.DataFrame()

# Select data for each year and concatenate with prices here 
for year in ['2013', '2014', '2015']:
    
    price_per_year = yahoo.loc['2013', ['price']].reset_index(drop=True)
    price_per_year.rename(columns={'price': '2013'}, inplace=True)
    prices = pd.concat([prices, price_per_year], axis=1)
    
    
    ```

## Number of Rows/Columns Viewed in Output

```python
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

```

## Get Size of Dataframe in Mb
The f thing is str.format, https://docs.python.org/3/library/string.html#formatstrings, in this case to two decimal places
```python
print(f"{(sys.getsizeof(df)/1024**2):.2f} Mb")
```

## Reduce Dataframe Size

__!!!The printout of column dtype transformations is edited out. So be careful if you use this.__

```python
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
        #if verbose and best_type is not None and best_type != str(col_type):
            #print(f"Column {col} converted from {col_type} to {best_type}")

    if verbose:
        end_mem = memory_usage_mb(df, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(f"Memory usage decreased from"
              f" {start_mem:.2f}MB to {end_mem:.2f}MB"
              f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")```
              
              
 Define memory_usage_mb : 
 ```python
def memory_usage_mb(df, *args, **kwargs):
    """Dataframe memory usage in MB. """
    return df.memory_usage(*args, **kwargs).sum()
```

## Drop columns

```python
s1.drop(columns=['Disc Number', 'Track Number', 'Added By'])
```

## Array Slices 

```python
a[start:stop]  # items start through stop-1
a[start:]      # items start through the rest of the array
a[:stop]       # items from the beginning through stop-1
a[:]           # a copy of the whole array
```

https://stackoverflow.com/questions/509211/understanding-slice-notation

## Create Datraframes from Lists/Arrays 



```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'last_expr_or_assign'

df = pd.DataFrame({'Playlist':["microhouse","microhouse","attlas","attlas"],"Track Name":["make a move","mango","ryat","further"],"Spotify Uri":["5nUS4bSN0cFZB0knxyM4LZ","2f8eSlsreAHHzJ5SPkpYLf","3McvalY1RDYczyDmixyAwQ","4qEUN1lON8UjnUiOZc39ID"],"Playlist Uri":["1d4gyZxan7lK9KqYU2EJ","1d4gyZxan7lK9KqYU2EJ","2CInjKguWauO29QB21Co","2CInjKguWauO29QB21Co"]})
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playlist</th>
      <th>Track Name</th>
      <th>Spotify Uri</th>
      <th>Playlist Uri</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>microhouse</td>
      <td>make a move</td>
      <td>5nUS4bSN0cFZB0knxyM4LZ</td>
      <td>1d4gyZxan7lK9KqYU2EJ</td>
    </tr>
    <tr>
      <td>1</td>
      <td>microhouse</td>
      <td>mango</td>
      <td>2f8eSlsreAHHzJ5SPkpYLf</td>
      <td>1d4gyZxan7lK9KqYU2EJ</td>
    </tr>
    <tr>
      <td>2</td>
      <td>attlas</td>
      <td>ryat</td>
      <td>3McvalY1RDYczyDmixyAwQ</td>
      <td>2CInjKguWauO29QB21Co</td>
    </tr>
    <tr>
      <td>3</td>
      <td>attlas</td>
      <td>further</td>
      <td>4qEUN1lON8UjnUiOZc39ID</td>
      <td>2CInjKguWauO29QB21Co</td>
    </tr>
  </tbody>
</table>
</div>



### Pandas category types

https://pbpython.com/pandas_dtypes_cat.html

### Groupby, aggregate, sort 
```python
df_matriz_2.groupby("sgt_cem").ing_bruto.agg("mean").nlargest()
df_matriz_2.groupby("ing_bruto").cem_2.agg({"cem_2_mean" : "mean"}).sort_index(ascending=False)

df[df.codmes.astype('str').str[0:4]=="2019"].groupby("codmes").ing_bruto.agg("max").sort_values(ascending=False)```

### Thousand separators & supress scientific notation 

```python
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.options.display.float_format = '{:,}'.format
```
So after having trouble rounding decimals to two places for float32, this worked. 
```python 
X  = round(X.astype(float),2)```

```python 
X.monto_ec = X.monto_ec.astype(int)```


__Get rid of scientific notation in numpy__:
```python 
np.set_printoptions(suppress=True)
```



### SKL: StandardScaler

Substracts by the mean and divides by the standard deviation. Only undestands numpy arrays.
- scaler= StandardScaler() - Assign the function to a variable
- scaler.fit(X) - Gives it the data and its creates estimates.
- scaler.mean_ 
- scaler.scale_ = Stddev

You can check, by manually find the mean and stddev for each of the numpy arrays. 
- np.std()
- np.mean()




```python
from sklearn.preprocessing import StandardScaler

import pandas as pd



X = [[0, 15],[1, -10]]

StandardScaler().fit(X).transform(X)
```




    array([[-1.,  1.],
           [ 1., -1.]])



### Scatter plots 
https://jakevdp.github.io/PythonDataScienceHandbook/04.02-simple-scatter-plots.html

### DF.plot()
```python 
X.plot(kind="scatter",x="monto_ec",y="tea_ec",marker="o",s=3,figsize= (18,15)) 
xplot = X.plot(x = "monto_ec", y = "tea_ec", style="o", ms=2, figsize= (18,15), legend = False)
xplot.set(xlabel="monto_ec",ylabel="tea_ec")
xplot
```
- S = size
- https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.scatter.html

### Plotly
```python 
import plotly as plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.express as px
px.scatter(X, x="monto_ec",y = "tea_ec")

```


### SKL: KMeans

Kmeans is a unsupervised machine learning algorithm which attempts to cluster. <br>
It starts by creating centroids randomly and then trying to reduce the distance to each data point. <br>
The tricky part is finding the right number of clusters.<br>
Inertia: Inertia is the sum of the squared distances between each training instance and its closest centroid:<br> 
"To select the best model, we will need a way to evaluate a K-Mean model's performance. Unfortunately, clustering is an unsupervised task, so we do not have the targets. But at least we can measure the distance between each instance and its centroid. This is the idea behind the inertia metric".<br>
K: Clusters <br>
As you can see, the inertia drops very quickly as we increase k up to 4, but then it
decreases much more slowly as we keep increasing k. This curve has roughly the
shape of an arm, and there is an “elbow” at k=4 so if we did not know better, it would
be a good choice: any lower value would be dramatic, while any higher value would
not help much, and we might just be splitting perfectly good clusters in half for no
good reason.
![caption](kmeans_elbow.png)

 

#### Optimal Number of Clusters: Plotting # clusters as function of inertia = Elbow Curve

The major drawback of the K-Means algorithm is that it
often gets stuck at local minima and the result is largely dependent on the
choice of the initial cluster centers.

This is just a visualisation of the the clusters are segmenting the data
```python
kmeans = KMeans(n_clusters=2).fit(X.values)
centroids = kmeans.cluster_centers_
print(centroids)

labels = kmeans.predict(X.values)
# Getting the cluster centers
C = kmeans.cluster_centers_
colores=['red','green']
asignar=[]
for row in labels:
    asignar.append(colores[row])
 
# Getting the values and plotting it
f1 = X['tea_ec'].values
f2 = X['monto_ec'].values
 
plt.scatter(f1, f2, c=asignar, s=70)
plt.show()                
``` 

### Numpy Reshape

-1 : Unkown dimension, numpy figure it out. 

So (-1,1) is saying convert this array, tensor, matrix, into a one column array/vector 
with however many rows it originally has. 

```python
x.reshape(-1,1)
```




### Manipulating tensors with Numpy

```python
my_slice = train_images[10:100, 0:28, 0:28]
my_slice.shape
(90, 28, 28)
```

### Dictionaries

```python 
word_index = imdb.get_word_index()
reverse_word_index = dict(
[(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
[reverse_word_index.get(i - 3, '?') for i in train_data[0]])
```

### Join
```python 
' '.join(
[reverse_word_index.get(i - 3, '?') 
```

### Warnings 
```python
import warnings
warnings.filterwarnings("ignore")
```

### Print every line of the cell 
```python 
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity= "all"
```

### Show what you're assigning
```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'last_expr_or_assign'
```
### Beep when done 
```python
from IPython.display import Audio

import numpy as np

wave = np.sin(8*np.pi*500*np.arange(10000*0.15)/10000)

Audio(wave, rate=10000, autoplay=True)
```
### See all session variables
```python
%whos
```
