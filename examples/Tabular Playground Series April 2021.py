import numpy as np

from pyutils import *
import seaborn as sns
import matplotlib.pyplot as plt

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import *
# from sklearn.metrics import *
# from sklearn.model_selection import *
# import lightgbm as lgb
desired_width = 150
pd.set_option('display.width', desired_width)
pd.set_option('max.columns', 12)
import dtale

# Load
read_data('/home/r/Downloads/tabular-playground-series-apr-2021.zip')
sample_submission, test, train = read_data('/home/r/Downloads/tabular-playground-series-apr-2021.zip', True)
sample_submission.shape, test.shape, train.shape

# EDA
dtale.show(train)
train
train.Survived.value_counts(normalize=True) * 100
describe_df(train)
(train.Age < 1).value_counts()
train[(train.Age < 1)].value_counts('Survived', normalize=True)
#Babies under the age of 1
pd.cut(train.loc[train.Age < 1, 'Age'] * 12, bins=5, right=True).value_counts(normalize=True).sort_index() * 100
pd.cut(train.loc[train.Age < 1, 'Age'], bins=5, right=True).value_counts().sort_index()
pd.cut(train.loc[train.Age < 1, 'Age']*12, bins=5, right=True).value_counts().sort_index()
train[train.Age.between(0.92, 1, inclusive = False)]
#Babies under the age of 1 who died
train.loc[(train.Age < 1) & (train.Survived ==0 )]
#Babies under the age of 1 who died by class
train.loc[(train.Age < 1) & (train.Survived ==0 )].value_counts(train.Pclass).sum()
#Dead by class global
train[train.Survived==0].value_counts(train.Pclass,normalize=True).sort_index()*100
#we got a 30& reduction in mortality by not being in the lowest class.
#Mortlity by embarked
#Plots
fig, ax = plt.subplots(figsize=(12,6))
sns.countplot(data=train, x='Embarked', hue='Survived', ax=ax)

train.loc[:,['Embarked','Survived']].value_counts().sort_index().unstack().plot.bar(figsize=(12,6),rot=0)
train.loc[:,['Embarked','Survived']].value_counts().sort_index().unstack(1)

sns.scatterplot(data=train, x="Age", y="Fare",hue='Embarked')
#Fare embarked
train.loc[:,['Embarked','Fare']].groupby('Embarked')

train.loc[:,['Embarked','Fare']].groupby('Embarked').describe().unstack(1)
train.loc[:,['Embarked','Survived']].value_counts(train.Survived)

train.Embarked.value_counts(normalize=True)*100
train.Embarked.value_counts()

describe_df(train)
train.Cabin
#FE
y= train.Survived

train_dropped = train.drop(columns= ['Cabin','Ticket','PassengerId','Survived','Name'],axis=1)


train_dropped_encoded= pd.concat([train_dropped.drop(['Embarked','Sex'], axis=1), encode(pd.DataFrame(train_dropped.loc[:,['Embarked','Sex']]),method='lbl')],axis=1)

train_dropped_encoded.loc[train_dropped_encoded.Embarked==3,'Embarked'] = np.NAN

describe_df(train_dropped_encoded)

#Any colmns with nas
na_cols = train_dropped_encoded[train_dropped_encoded.columns[train_dropped_encoded.isna().any()]]

describe_df(train_dropped_encoded)



import miceforest as mf
# Create kernel.
kds = mf.KernelDataSet(
  na_cols,
  save_all_iterations=True,
  random_state=1991
)
# Run the MICE algorithm for 3 iterations
kds.mice(3)

# Return the completed kernel data
completed_data = kds.complete_data()

describe_df(completed_data)
describe_df(na_cols)

train_dropped_encoded_nonulls = pd.concat([completed_data,train_dropped_encoded.drop(list(completed_data.columns.values),axis=1)],axis=1)



# sibsp = # of siblings / spouses aboard the Titanic
# parch =  # of parents / children aboard the Titanic
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

# Variable Notes
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
#
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
#
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
#
# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

# ticket	Ticket number
# fare	Passenger fare