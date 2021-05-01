import matplotlib.pyplot as plt

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
# data = pd.concat([data.iloc[:,:3], train.churn.map({"no": 0 , "yes":1})],axis=1)
data = pd.melt(data, id_vars="churn",
               var_name="features",
               value_name='value')
# boxplot
# plt.figure(figsize=(10,10))
# sns.boxplot(x="features", y="value", hue="churn", data=data)
# plt.xticks(rotation=20)

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

attend = sns.load_dataset("attention").query("subject <= 12")

plt.figure()
g = sns.FacetGrid(data, col="features", col_wrap=5, ylim=(0, 0.8), height=2)
g.map(sns.kdeplot, "value", color='black', shade=True)
#g.add_legend()
plt.show()

plt.figure()
sns.kdeplot(data=data, x="value", color='black', shade=True, legend=True)
plt.show()

# swarm plot
# sns.set(style="whitegrid", palette="muted")
# plt.figure(figsize=(10,10))
# sns.swarmplot(x="features", y="value", hue="churn", data=data)
# plt.xticks(rotation=90)


plot_univariate_classification(tr, "churn")

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
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
