import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from pyutils import *
import zipfile
import time
from datetime import timedelta



path = '/home/r/Downloads/interbank20.zip'
path2 = '/home/r/Downloads/intro-ml-project-main.zip'
path3 = '/home/r/Downloads/house-prices-advanced-regression-techniques'

csvs = ['/home/r/Downloads/house-prices-advanced-regression-techniques/train.csv',
        '/home/r/Downloads/house-prices-advanced-regression-techniques/test.csv',
        '/home/r/Downloads/house-prices-advanced-regression-techniques/sample_submission.csv']

read_data(csvs)


def read_data(path_ending_with_filename=None, return_df=False, method=None):
    """
    Reads single csv or list of csvs or csvs in zip.

    Available methods:
        'dt' = Datatable fread

    TODO: Add to read methods. i.e., parquet, pickle, arrow, etc.
    """
    dt.options.progress.enabled = True
    if isinstance(path_ending_with_filename, str):
        if path_ending_with_filename.endswith('.zip'):
            zf = zipfile.ZipFile(path_ending_with_filename)
            files = zf.namelist()
            if return_df:
                dfs = {}
                start_time = time.monotonic()
                for x in files:
                    if x.endswith('.csv'):
                        if method == 'dt':
                            dfs["{0}".format(re.findall("\w+(?=\.)", x)[0])] = dt.fread(zf.open(x)).to_pandas()
                        else:
                            dfs["{0}".format(re.findall("\w+(?=\.)", x)[0])] = pd.read_csv(zf.open(x))
                end_time = time.monotonic()
                print(timedelta(seconds=end_time - start_time))

                keys = list(dfs.keys())
                values = list(dfs.values())
                for i in enumerate(dfs):
                    print(i[1], ":", values[i[0]].shape)

                return dfs.values()
            else:
                filelist = zf.filelist
                csv_file_names = [format(re.findall("\w+(?=\.)", zf.namelist()[i])[0]) for i in
                                  range(len(zf.namelist())) if zf.namelist()[i].endswith('.csv')]
                file_pos = [i for i, x in enumerate(zf.namelist()) if x.endswith('.csv')]
                uncompressed_dir = [f"{(zf.filelist[i].file_size / 1024 ** 2):.2f} Mb" for i in file_pos]
                compressed = [f"{(zf.filelist[i].compress_size / 1024 ** 2):.2f} Mb" for i in file_pos]

                print(pd.concat([pd.Series(csv_file_names), pd.Series(uncompressed_dir), pd.Series(compressed)], axis=1,
                                keys=["file_names", "uncompressed", "compressed"]))
                print()
                print(*csv_file_names, sep=",")


        else:
            # SINGLE FILE
            if path_ending_with_filename.endswith(".csv"):
                df_name = re.findall("\w+(?=\.)", path_ending_with_filename)[0]
                if method == 'dt':
                    df = dt.fread(path_ending_with_filename)
                    df = df.to_pandas()
                else:
                    df = pd.read_csv(path_ending_with_filename)
                if return_df:
                    return df
                else:
                    print(df_name, df.shape)
            else:
                # CSVS IN DIRECTORY
                dfs = {}
                os.chdir(path_ending_with_filename)
                for x in os.listdir(path_ending_with_filename):
                    if x.endswith('.csv'):
                        if method == 'dt':
                            dfs["{0}".format(re.findall("\w+(?=\.)", x)[0])] = dt.fread(x).to_pandas()
                        else:
                            dfs["{0}".format(re.findall("\w+(?=\.)", x)[0])] = pd.read_csv(x)
                keys = list(dfs.keys())
                values = list(dfs.values())
                if return_df:
                    return dfs.values()
                else:

                    uncompressed_dir = [f"{(sys.getsizeof(dfs[i]) / 1024 ** 2):.2f} Mb" for i in dfs]

                    print(pd.concat([pd.Series(keys), pd.Series(uncompressed_dir)], axis=1,
                                    keys=["file_names", "uncompressed"]))
                    print()
                    print(*keys, sep=",")



    else:
        # LIST OF CSV FILES
        dfs = {}
        for x in path_ending_with_filename:
            if x.endswith('.csv'):
                if method == 'dt':
                    dfs["{0}".format(re.findall("\w+(?=\.)", x)[0])] = dt.fread(x).to_pandas()
                else:
                    dfs["{0}".format(re.findall("\w+(?=\.)", x)[0])] = pd.read_csv(x)
        keys = list(dfs.keys())
        values = list(dfs.values())
        if return_df:
            return dfs.values()
        else:
            for i in enumerate(dfs):
                print(i[1], " ", "=", " ", "(", f"{values[i[0]].shape[0]:,}", ":", f"{values[i[0]].shape[1]:,}", ")",
                      sep="")

            print(str(",".join(keys)))



read_data(path)
censo_test, censo_train, productos, rcc_test, rcc_train, sample_submission, se_test, se_train, sunat_test, sunat_train, y_train = read_data(path, True, "dt")

def all_dfs(list_of_dfs, list_of_dfs_with_single_quotes):
    """Example:
    s = private_test_data, question_meta, student_meta, subject_meta, test_data, train_data, valid_data
s2 = 'private_test_data, question_meta, student_meta, subject_meta, test_data, train_data, valid_data'"""
    dic = {}
    dfs= list_of_dfs
    s2 = list_of_dfs_with_single_quotes.split(",")
    for i, p in enumerate(dfs):
        dic["{0}".format(s2[i])] = p
    keys = list(dic.keys())
    values = list(dic.values())
    for i in enumerate(dic):
        print(i[1], " ", "=", " ", "(", f"{values[i[0]].shape[0]:,}", ":", f"{values[i[0]].shape[1]:,}", ")",
              sep="")

#        print(str(",".join(keys)))
all_dfs(s,s2)


private_test_data, question_meta, student_meta, subject_meta, test_data, train_data, valid_data = read_data(path2, True,                                                                                                           'dt')

s = private_test_data, question_meta, student_meta, subject_meta, test_data, train_data, valid_data
s2 = 'private_test_data, question_meta, student_meta, subject_meta, test_data, train_data, valid_data'

all_dfs(s,s2)



dfs =[private_test_data, question_meta, student_meta, subject_meta, test_data, train_data, valid_data]


for i,p in enumerate(dfs):
    dic["{0}".format(s2[i])] = p


for i in dfs:
    dic["{0}".format()= i]


dfs= {"censo_test":censo_test}
len(dfs)
for i in dfs:
    print()

dfs[0]

    print(i, ":", dfs[i[0]].shape)


directory(path)

os.chdir(re.findall("^(.*[\\\/])", path3)[0])
os.listdir(path3)

read_data(path3)

# TODO:  1. Add timer to data.table fread. 2. Sort out the functions in load so they are coherent and can be used in conjunction. 3.


read_data(path)
read_data(path3)

private_test_data, question_meta, student_meta, subject_meta, test_data, train_data, valid_data = read_data(path2, True)

private_test_data.shape, question_meta.shape, student_meta.shape, subject_meta.shape, test_data.shape, train_data.shape, valid_data.shape

for i in enumerate(dfs):
    print(i[1], ":", values[i[0]].shape)
print(str(",".join(keys)))
print(str(".shape,".join(keys)), ".shape", sep='')

zf = zipfile.ZipFile(path2)

filelist = zf.filelist

csv_file_names = [format(re.findall("\w+(?=\.)", zf.namelist()[i])[0]) for i in range(len(zf.namelist())) if
                  zf.namelist()[i].endswith('.csv')]

file_pos = [i for i, x in enumerate(zf.namelist()) if x.endswith('.csv')]

uncompressed = [f"{(zf.filelist[i].file_size / 1024 ** 2):.2f} Mb" for i in file_pos]

compressed = [f"{(zf.filelist[i].compress_size / 1024 ** 2):.2f} Mb" for i in file_pos]

pd.concat([pd.Series(csv_file_names), pd.Series(uncompressed), pd.Series(compressed)], axis=1,
          keys=["file_names", "uncompressed", "compressed"])

print(*csv_file_names, sep=",")

zipfile.ZipFile(path).infolist()[1]

print(f"{(110209469 / 1024 ** 2):.2f} Mb")
print(f"{(5824159 / 1024 ** 2):.2f} Mb")

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
# g.add_legend()
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
