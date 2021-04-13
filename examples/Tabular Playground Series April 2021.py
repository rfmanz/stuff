from pyutils import *
import seaborn as sns
import matplotlib.pyplot as plt
import miceforest as mf

from sklearn.model_selection import *
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from lightgbm import LGBMClassifier
import optuna
import optuna.integration.lightgbm as lgb

import warnings


# pd.options
desired_width = 150
pd.set_option('display.width', desired_width)
pd.set_option('max.columns', 20)

warnings.filterwarnings('ignore')


# Load
read_data('/home/r/Downloads/tabular-playground-series-apr-2021.zip')
sample_submission, test, train = read_data('/home/r/Downloads/tabular-playground-series-apr-2021.zip', True)
sample_submission.shape, test.shape, train.shape

#0.5 eda

train.Name.nunique()
train.iloc[:4].T
describe_df(train)
train[].value_counts(train.Survived)


# EDA
# train.Survived.value_counts(normalize=True) * 100
# sns.countplot()
# describe_df(train)
# (train.Age < 1).value_counts()
# train[(train.Age < 1)].value_counts('Survived', normalize=True)
# # Babies under the age of 1
# pd.cut(train.loc[train.Age < 1, 'Age'] * 12, bins=5, right=True).value_counts(normalize=True).sort_index() * 100
# pd.cut(train.loc[train.Age < 1, 'Age'], bins=5, right=True).value_counts().sort_index()
# pd.cut(train.loc[train.Age < 1, 'Age'] * 12, bins=5, right=True).value_counts().sort_index()
# train[train.Age.between(0.92, 1, inclusive=False)]
# # Babies under the age of 1 who died
# train.loc[(train.Age < 1) & (train.Survived == 0)]
# # Babies under the age of 1 who died by class
# train.loc[(train.Age < 1) & (train.Survived == 0)].value_counts(train.Pclass).sum()
# # Dead by class global
# train[train.Survived == 0].value_counts(train.Pclass, normalize=True).sort_index() * 100
# # we got a 30& reduction in mortality by not being in the lowest class.
# # Mortlity by embarked
# # Plots
# fig, ax = plt.subplots(figsize=(12, 6))
# sns.countplot(data=train, x='Embarked', hue='Survived', ax=ax)
#
# train.loc[:, ['Embarked', 'Survived']].value_counts().sort_index().unstack().plot.bar(figsize=(12, 6), rot=0)
# train.loc[:, ['Embarked', 'Survived']].value_counts().sort_index().unstack(1)
#
# sns.scatterplot(data=train, x="Age", y="Fare", hue='Embarked')
# # Fare embarked
# train.loc[:, ['Embarked', 'Fare']].groupby('Embarked')
#
# train.loc[:, ['Embarked', 'Fare']].groupby('Embarked').describe().unstack(1)
# train.loc[:, ['Embarked', 'Survived']].value_counts(train.Survived)
#
# train.Embarked.value_counts(normalize=True) * 100
# train.Embarked.value_counts()
#
# # Comparing train & test
# # Age -testvstrain
# ax = sns.distplot(train.Age, hist=False, label="Train", color='olive', kde=True)
# ax = sns.distplot(test.Age, hist=False, label="Test", color='blue', kde=True)
#
# l1 = ax.lines[0]
# l2 = ax.lines[1]
#
# x1 = l1.get_xydata()[:, 0]
# y1 = l1.get_xydata()[:, 1]
# x2 = l2.get_xydata()[:, 0]
# y2 = l2.get_xydata()[:, 1]
# ax.fill_between(x1, y1, color='olive', alpha=0.3)
# ax.fill_between(x2, y2, color="blue", alpha=0.3)
# plt.legend()
# plt.show(block=False)
# # FAre tst/trn
# ax = sns.distplot(train.Fare, hist=False, label="Train", color='olive', kde=True)
# ax = sns.distplot(test.Fare, hist=False, label="Test", color='blue', kde=True)
#
# l1 = ax.lines[0]
# l2 = ax.lines[1]
#
# x1 = l1.get_xydata()[:, 0]
# y1 = l1.get_xydata()[:, 1]
# x2 = l2.get_xydata()[:, 0]
# y2 = l2.get_xydata()[:, 1]
# ax.fill_between(x1, y1, color='olive', alpha=0.3)
# ax.fill_between(x2, y2, color="blue", alpha=0.3)
# plt.legend()
# plt.show(block=False)
# # Scatterplot to visualize outliers
# train.Age.plot(style='.')
# sns.scatterplot(train.loc[(train.Age > 80), 'Age'], train[(train.Age > 80)].index, marker='x', s=20)
# sns.scatterplot(train.loc[(train.Age > 80), 'Age'], train[(train.Age > 80)].index)
#
# sns.distplot(train.Age, hist=True, color='black')
# sns.kdeplot(train.Age, color='black', shade=True)


# FE
y = train.Survived


def converter(x):
    c, n = '', ''
    x = str(x).replace('.', '').replace('/', '').replace(' ', '')
    for i in x:
        if i.isnumeric():
            n += i
        else:
            c += i
    if n != '':
        return c, int(n)
    return c, np.nan


def create_extra_features(data):
    data['Ticket_type'] = data['Ticket'].map(lambda x: converter(x)[0])
    data['Cabin_type'] = data['Cabin'].map(lambda x: converter(x)[0])
    data['FamilySize'] = data['SibSp'] + data['Parch']
    data['isAlone'] = data['FamilySize'].apply(lambda x: 1 if x == 0 else 0)
    data['age*fare'] = data.Age * data.Fare
    return data


train = create_extra_features(train)
test = create_extra_features(test)

train_dropped = train.drop(columns=['Cabin', 'Ticket', 'PassengerId', 'Survived', 'Name'], axis=1)
test_dropped = test.drop(columns=['Cabin', 'Ticket', 'PassengerId', 'Name'], axis=1)

cat = train_dropped.select_dtypes('object').columns
cont = train_dropped.select_dtypes('number').columns


def encodethis(data):
    data = pd.concat([data[cont], encode(data[cat], method='lbl')], axis=1)
    return data


train_dropped_encoded = encodethis(train_dropped)
test_dropped_encoded = encodethis(test_dropped)


train_dropped_encoded.loc[train_dropped_encoded.Embarked == 3, 'Embarked'] = np.NAN
test_dropped_encoded.loc[test_dropped_encoded.Embarked == 3, 'Embarked'] = np.NAN


results = []
for i in train_dropped_encoded, test_dropped_encoded:
    na_cols = i[i.columns[i.isna().any()]]

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

    t = pd.concat([completed_data, i.drop(list(completed_data.columns.values), axis=1)], axis=1)
    results.append(t)

train_dropped_encoded_nonulls = results[0]
test_dropped_encoded_nonulls = results[1]



train_dropped_encoded_nonulls = pd.concat([train_dropped_encoded_nonulls.drop(['Ticket_type','Cabin_type'],axis=1),encode(train_dropped_encoded_nonulls[['Ticket_type','Cabin_type']].astype('object'))],axis=1)
test_dropped_encoded_nonulls = pd.concat([test_dropped_encoded_nonulls.drop(['Ticket_type','Cabin_type'],axis=1),encode(test_dropped_encoded_nonulls[['Ticket_type','Cabin_type']].astype('object'))],axis=1)

train_dropped_encoded_nonulls = pd.DataFrame(MinMaxScaler().fit_transform(train_dropped_encoded_nonulls),columns= train_dropped_encoded_nonulls.columns, index= train_dropped_encoded_nonulls.index)

test_dropped_encoded_nonulls = pd.DataFrame(MinMaxScaler().fit_transform(test_dropped_encoded_nonulls),columns= test_dropped_encoded_nonulls.columns, index= test_dropped_encoded_nonulls.index)


# train_dropped_encoded_nonulls = standard_scaler(train_dropped_encoded_nonulls)
# test_dropped_encoded_nonulls = standard_scaler(test_dropped_encoded_nonulls)


 train_dropped_encoded_nonulls.iloc[:5].T
#describe_df(train_dropped_encoded_nonulls)
# describe_df(test_dropped_encoded_nonulls)
correlated(train_dropped_encoded_nonulls,0.8)
# train_dropped_encoded_nonulls = correlated(train_dropped_encoded_nonulls,0.8,True)
# test_dropped_encoded_nonulls = correlated(test_dropped_encoded_nonulls,0.8,True)

#CV

x_train, x_test, y_train, y_test = train_test_split(train_dropped_encoded_nonulls, y, test_size=.2)


#OPT .5
#kf = KFold(n_splits=5)
kf = StratifiedKFold(n_splits=10, shuffle=True)


params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }


study_tuner = optuna.create_study(direction='maximize')
#x = train_dropped_encoded_nonulls

dtrain = lgb.Dataset(x, label=y)

# Run optuna LightGBMTunerCV tuning of LightGBM with cross-validation
#pruning_callback = optuna.integration.LightGBMPruningCallback(study_tuner, "auc")

tuner = lgb.LightGBMTunerCV(params,
                            dtrain,
                            study=study_tuner,
                            early_stopping_rounds=250,
                            folds=kf,
                            num_boost_round=1500)
                            #callbacks=pruning_callback

tuner.run()
print(tuner.best_params)
# Classification error
print(tuner.best_score)


tmp_best_params = tuner.best_params
if tmp_best_params['feature_fraction']==1:
    tmp_best_params['feature_fraction']=1.0-1e-9
if tmp_best_params['feature_fraction']==0:
    tmp_best_params['feature_fraction']=1e-9
if tmp_best_params['bagging_fraction']==1:
    tmp_best_params['bagging_fraction']=1.0-1e-9
if tmp_best_params['bagging_fraction']==0:
    tmp_best_params['bagging_fraction']=1e-9


# import lightgbm as lgb
# dtrain = lgb.Dataset(x, label=y)
# def objective(trial):
#     params = {
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0),
#         'num_leaves': trial.suggest_int('num_leaves', 11, 333),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#         'max_depth': trial.suggest_int('max_depth', 5, 20),
#         'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.005, 0.1]),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5),
#         'n_estimators': trial.suggest_int('n_estimators', 50, 3000),
#         'random_state': 42,
#         'boosting_type': 'gbdt',
#         'metric': 'AUC',
#         'device': 'cpu',
#         'feature_pre_filter' : 'false'
#
#     }
#
# lgbcv = lgb.cv(tmp_best_params,
#                dtrain,
#                folds=kf,
#                verbose_eval=False,
#                early_stopping_rounds=250,
#                num_boost_round=1500)
#
# # Run LightGBM for the hyperparameter values
# #pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
#     lgbcv = lgb.cv(params,
#                    dtrain,
#                    folds=kf,
#                    verbose_eval=False,
#                    early_stopping_rounds=250,
#                    num_boost_round=1500)
#                    #callbacks= pruning_callback)

#
# optuna.logging.set_verbosity(optuna.logging.WARNING)
# study = optuna.create_study(direction='maximize')
# study.enqueue_trial(tmp_best_params)
# study.optimize(objective, n_trials=30)
# print('Best trial:', study.best_trial.params)
# print('Best value:', study.best_value)


# OPT
def objective(trial):
    params = {
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 11, 333),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.005, 0.1]),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 50, 3000),
        'random_state': 42,
        'boosting_type': 'gbdt',
        'metric': 'AUC',
        'device': 'cpu'
    }

    model = LGBMClassifier(**params)
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=222, verbose=False)
    y_pred = model.predict_proba(x_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred)

    return roc_auc
from optuna.pruners import SuccessiveHalvingPruner

study = optuna.create_study(direction='maximize',pruner=SuccessiveHalvingPruner())
study.enqueue_trial(tmp_best_params)
study.optimize(objective, n_trials=100)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
print('Best value:', study.best_value)

optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)


paramsLGBM = study.best_trial.params
#paramsLGBM = tuner.best_params
paramsLGBM['boosting_type'] = 'gbdt'
paramsLGBM['metric'] = 'AUC'
paramsLGBM['random_state'] = 42
paramsLGBM['objective'] = 'binary'

{'reg_alpha': 1.7756323120719368,
 'reg_lambda': 1.1329669604585568,
 'num_leaves': 32,
 'min_child_samples': 74,
 'lambda_l1': 0.10778419855175325,
 'feature_fraction': 0.4199498862570688,
 'bagging_fraction': 0.9192428145485326,
 'bagging_freq': 1,
 'max_depth': 16,
 'learning_rate': 0.02,
 'colsample_bytree': 0.25638115462185734,
 'n_estimators': 731}


# MDL

kf = KFold(n_splits=10, shuffle=True, random_state=42)

x = train_dropped_encoded_nonulls
y = train.Survived
auc = []
preds = np.zeros(test_dropped_encoded_nonulls.shape[0])
n=0   


for fold, (trn_idx, val_idx) in enumerate(kf.split(x, y)):
    print(f"===== FOLD {fold+1} =====")
    x_train, x_val = x.iloc[trn_idx], x.iloc[val_idx]
    y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    model = LGBMClassifier(**paramsLGBM)

    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='auc', verbose=-1,early_stopping_rounds=500)

     preds += model.predict_proba(test_dropped_encoded_nonulls)[:, 1] / kf.n_splits

    auc.append(roc_auc_score(y_val, model.predict_proba(x_val)[:, 1]))
    
np.mean(auc)
#roc mena
#0.859017


sample_submission.iloc[:, 1] = np.where(preds > 0.5, 1, 0)
sample_submission

sample_submission.to_csv('~/Downloads/tabular_playground_april_9.csv', index=False)


predictions = model.predict(test_dropped_encoded_nonulls,num_iteration=model.best_iteration_)





import lightgbm
plt.rcParams["figure.figsize"] = (6, 5)
lightgbm.plot_importance(model, max_num_features=16, height=.9)






RobustScalar

# INFO ON THE DATA

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

#catboost
#xgboost
#rf
#SVM
#ridge
#neural net

http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/