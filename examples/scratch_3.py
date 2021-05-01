import matplotlib.pyplot as plt
import pandas as pd

from pyutils import *

path = '/home/r/Downloads/customer-churn-prediction-2020.zip'
read_data(path)
sampleSubmission, test, train = read_data(path, True)
sampleSubmission.shape, test.shape, train.shape
tr = train
continuous_cols = list(tr.select_dtypes("number").columns)

violin_plot(train, "Survived")
plot_univariate_classification(train, "Survived")

data = pd.DataFrame(StandardScaler().fit_transform(tr[continuous_cols]), columns=tr[continuous_cols].columns,
                    index=tr[continuous_cols].index)
data = pd.concat([data, train.churn.map({"no": 0, "yes": 1})], axis=1)

data = pd.melt(data, id_vars="churn",
               var_name="features",
               value_name='value')
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

from pyutils.eda import _make_subplots

fig, axes = _make_subplots(n_plots=len(continuous_cols), row_height=2)
for i, (ind, ax) in enumerate(zip(continuous_cols, axes.ravel())):
    sns.kdeplot(tr[continuous_cols[i]], color='black', shade=True, legend=True, ax=ax)
for j in range(i + 1, axes.size):
    # turn off axis if we didn't fill last row
    axes.ravel()[j].set_axis_off()
plt.show()


def plot_density_numerical(df):
    continuous_cols = list(df.select_dtypes("number").columns)
    fig, axes = _make_subplots(n_plots=len(continuous_cols), row_height=2)
    for i, (ind, ax) in enumerate(zip(continuous_cols, axes.ravel())):
        sns.kdeplot(df[continuous_cols[i]], color='black', shade=True, legend=True, ax=ax)
    for j in range(i + 1, axes.size):
        # turn off axis if we didn't fill last row
        axes.ravel()[j].set_axis_off()
    return plt.show()


plot_density_numerical(train)

plt.figure()
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

tr.dtypes
plt.figure()
sns.histplot(x=tr.international_plan, data=tr, color='green')
plt.show()
plt.figure()
categorical_cols = list(tr.select_dtypes("object").columns)
fig, axes = _make_subplots(n_plots=len(categorical_cols), row_height=2)
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
for i, (ind, ax) in enumerate(zip(categorical_cols, axes.ravel())):
    ax = sns.histplot(x=tr[categorical_cols[i]], data=tr, color='green', ax=ax)
for j in range(i + 1, axes.size):
    axes.ravel()[j].set_axis_off()
#plt.xticks(rotation='vertical')
plt.show()

# interesting to use:
# segmented pie chart
# cumulative graphs

# when you're done with graphing : Feature's
# https://featuretools.alteryx.com/en/stable/
# https://github.com/alteryx/featuretools
