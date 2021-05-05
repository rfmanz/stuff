import matplotlib.pyplot as plt
import pandas as pd

from pyutils import *

path = '/home/r/Downloads/customer-churn-prediction-2020.zip'
read_data(path)
sampleSubmission, test, train = read_data(path, True)
sampleSubmission.shape, test.shape, train.shape
tr = train
continuous_cols = list(tr.select_dtypes("number").columns)

data = pd.DataFrame(StandardScaler().fit_transform(tr[continuous_cols]), columns=tr[continuous_cols].columns,
                    index=tr[continuous_cols].index)
data = pd.concat([data, train.churn.map({"no": 0, "yes": 1})], axis=1)

data = pd.melt(data, id_vars="churn",
               var_name="features",
               value_name='value')
import plotly.io as pio
pio.renderers.default = "browser"

import plotly.express as px
fig = px.violin(data,)
fig.show()

# boxplot
plt.figure(figsize=(40, 40))
sns.boxplot(x="features", y="value", hue="churn", data=data)
plt.xticks(rotation=20)

# violin plot
# sns.set_theme(style="whitegrid")
sns.reset_orig()
plt.figure(figsize=(10, 10))
ax = sns.violinplot(x="features", y="value", hue="churn", data=data, split=True, inner="quartile")
for i in range(len(np.unique(data["features"])) - 1):
    ax.axvline(i + 0.5, color='grey', lw=1)

# sns.violinplot(x="value", y="features", hue="churn", data=data, split=True, inner="quartile")
# ax = sns.stripplot(x="features", y="value", data=data,size= 0.8,color = "black")
plt.xticks(rotation=20)
plt.show()

# global density

attend = sns.load_dataset("attention").query("subject <= 12")

plt.figure()
g = sns.FacetGrid(data, col="features", col_wrap=5, ylim=(0, 0.8), height=2)
g.map(sns.kdeplot, "value", color='black', shade=True)
# g.add_legend()
plt.show()

plt.figure()
sns.kdeplot(data=data, x="value", color='black', shade=True, legend=True)
plt.show()


fig, axes = make_subplots(n_plots=len(continuous_cols), row_height=2)
for i, (ind, ax) in enumerate(zip(continuous_cols, axes.ravel())):
    sns.kdeplot(tr[continuous_cols[i]], color='black', shade=True, legend=True, ax=ax)
for j in range(i + 1, axes.size):
    # turn off axis if we didn't fill last row
    axes.ravel()[j].set_axis_off()
plt.show()


def plot_density_numerical(df):
    continuous_cols = list(df.select_dtypes("number").columns)
    fig, axes = make_subplots(n_plots=len(continuous_cols), row_height=2)
    for i, (ind, ax) in enumerate(zip(continuous_cols, axes.ravel())):
        sns.kdeplot(df[continuous_cols[i]], color='black', shade=True, legend=True, ax=ax)
    for j in range(i + 1, axes.size):
        # turn off axis if we didn't fill last row
        axes.ravel()[j].set_axis_off()
    return plt.show()


plot_density_numerical(train)

plt.figure()
fig =
plt.show()
tr.Survived = tr.Survived.astype('object')
plot_univariate_classification(tr, "churn")

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
tr.plot.density()
tr.hist()
plt.show()

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(15, 15))
fig.suptitle('Distribution of Continuous Features (cont0-cont10)', fontsize=16)

for idx, col in enumerate(range(len(continuous_cols))):
    i, j = (idx // 4, idx % 4)
    sns.kdeplot(tr[continuous_cols[i]], color="blue", shade=True, ax=ax[i, j])
# label="%1.1f"%(tr[continuous_cols[i]].skew()),)

fig.delaxes(ax[2, 3])
plt.tight_layout()
plt.show()

# -----


# ----

# plot_univariate_classification_categorical
plot_univariate_classification(tr, "churn")

fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(15, 15))
fig.suptitle('Distribution of Categorical Features with respect to Targets(cat0-cat18)', fontsize=16)

catCols_s = ['cat0', 'cat1', 'cat2', 'cat3', 'cat4', 'cat6', 'cat9', 'cat11', 'cat12', 'cat13', 'cat14', 'cat15',
             'cat16', 'cat17', 'cat18']
for idx, col in enumerate(train_file[catCols_s]):
    i, j = (idx // 4, idx % 4)
    f1 = sns.histplot(x=col, data=train_file[train_file['target'].astype(int) == 0], ax=ax[i, j], color='green',
                      label='Target-0')
    f2 = sns.histplot(x=col, data=train_file[train_file['target'].astype(int) == 1], ax=ax[i, j], color='yellow',
                      label='Target-1')
    f1 = f1.legend(loc="best")
    f2 = f2.legend(loc="best")

fig.delaxes(ax[3, 3])
plt.tight_layout()
plt.show()

sns.countplot(tr.state)

#grid = plt.GridSpec( nrows=2, ncols=1, wspace=0.4, hspace=0.3)
grid = plt.GridSpec( nrows=2, ncols=1)
plt.grid()
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2]);
plt.show()
categorical_cols = list(tr.select_dtypes("object").columns)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
# fig, axes = make_subplots(n_plots=len(categorical_cols), row_height=2)

for i, (ind, ax) in enumerate(zip(categorical_cols, axes.ravel())):
    ax = sns.histplot(x=tr[categorical_cols[i]], data=tr, color='green', ax=ax)
    ax.set_xticklabels(labels=tr[categorical_cols[i]].value_counts().index.values, rotation=90)

# fig.delaxes(ax[3, 3])
# plt.xticks(rotate=90)
# plt.tight_layout()
plt.show()
# ---
categorical_cols = list(train.select_dtypes("object").columns)
fig, axes = make_subplots(n_plots=len(categorical_cols), row_height=2)
for i, (ind, ax) in enumerate(zip(categorical_cols, axes.ravel())):
    ax = sns.histplot(x=train[categorical_cols[i]], data=train, color='green', ax=ax)
    ax=  ax.set_xticklabels(labels=train[categorical_cols[i]].value_counts().index.values, rotation=90)

for j in range(i + 1, axes.size):
    axes.ravel()[j].set_axis_off()
    # plt.xticks(rotate=90)
plt.show()

categorical_cols

tr.international_plan.value_counts().index.values


def plot_bar_single_column(df):
    plt.figure()
    ax = sns.histplot(df)
    ax.set_xticklabels(labels=df.value_counts().index.values, rotation=90)
    return plt.show()


plot_bar_single_column(tr.state)
import plotly.express as px


import plotly.io as pio
pio.renderers.default = "browser"

fig = px.histogram(tr.state)
fig.show()

import dtale

dtale.show(tr)
from pandas_profiling import ProfileReport

profile = ProfileReport(tr, explorative=True)
profile

import plotly.graph_objs as go

s = train[~pd.isnull(train['Ticket_type'])]['Ticket_type']
chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])

plt.figure()
sns.barplot(y="labels", x="data", data=chart)
plt.show()

plt.figure()
sns.kdeplot(tr.number_customer_service_calls,color="black",shade="gray")
plt.show()

plt.figure()
ax = sns.barplot(y= "labels",x = "data",data = chart)
#ax.set_xlabel('labels')
plt.xticks(rotation = 90 )
plt.show()

plt.barh(chart.labels, chart.data)
chart.plot.barh(figsize=(10,20))
plt.show()

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency')]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'title': {'text': 'state Value Counts'},
    'xaxis': {'title': {'text': 'state'}},
    'yaxis': {'title': {'text': 'Frequency'}}
}))

# https://plotly.com/python/statistical-charts/

fig = figure
fig.show()

plt.figure()
ax = sns.histplot(tr[["state"]].sort_values)
ax.set_xticklabels(labels=tr.state, rotation=90)
# plt.xticks(rotoate=90)
plt.show()
# interesting to use:
# segmented pie chart
# cumulative graphs

# when you're done with graphing : Feature's
# https://featuretools.alteryx.com/en/stable/
# https://github.com/alteryx/featuretools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker

# Some random data
dfWIM = pd.DataFrame({'AXLES': np.random.normal(8, 2, 5000).astype(int)})
dfWIM
ncount = len(dfWIM)

plt.figure(figsize=(12,8))
ax = sns.barplot(x= "labels",y = "data",data = chart)
ncount = len(tr.state)
#plt.title('Distribution of Truck Configurations')
#plt.xlabel('Number of Axles')

# Make twin axis
ax2=ax.twinx()

# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()

# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Frequency [%]')

for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y),
            ha='center', va='bottom') # set the alignment of the text

# Use a LinearLocator to ensure the correct number of ticks
ax.yaxis.set_major_locator(ticker.LinearLocator(11))

# Fix the frequency range to 0-100
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)

# And use a MultipleLocator to ensure a tick spacing of 10
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

# Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
ax2.grid(None)
plt.show()

