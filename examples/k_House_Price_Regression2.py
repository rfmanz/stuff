    
import numpy as np
import pandas as pd
import plotly.express as px
import sys
import matplotlib.pyplot as plt

import lightgbm as lgb 
from sklearn.compose import TransformedTargetRegressor

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# options
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.max_rows = 30
pd.options.display.max_columns = 20

# Functions

def load_this_shit():
    global y
    train = pd.read_csv('/home/r/Downloads/train.csv')
    test = pd.read_csv('/home/r/Downloads/test.csv')
    y = train.SalePrice
    full_df = pd.concat([train.drop(columns= 'SalePrice'),test])
    return full_df

def handle_missing(features):
    # the data description states that NA refers to typical ('Typ') values
    features['Functional'] = features['Functional'].fillna('Typ')
    # Replace the missing values in each of the columns below with their mode
    features['Electrical'] = features['Electrical'].fillna("SBrkr")
    features['KitchenQual'] = features['KitchenQual'].fillna("TA")
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    # the data description stats that NA refers to "No Pool"

    #features["PoolQC"] = features["PoolQC"].fillna("None")
    # Replacing the missing values with 0, since no garage = no cars in garage
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)
    # Replacing the missing values with None
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')
    # NaN values for these categorical basement features, means there's no basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None');

    # Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    objects = []
    for i in features.columns:
        if features[i].dtype == object:
            objects.append(i)
    features.update(features[objects].fillna('None'))

    # And we do the same thing for numerical features, but this time with 0s
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numeric.append(i)
    features.update(features[numeric].fillna(0))
    return features

def encode(df, method="dumy"):
    cols = df.select_dtypes('object').columns

    if method == "lbl":
        for i in cols:
            df[i] = LabelEncoder().fit_transform(df[i].astype(str))
        return df

    if method == "dumy":
        encoded = pd.get_dummies(df[cols])
        df = df.drop(columns=cols, axis=1)
        df = pd.concat([df, encoded], axis=1)
        return df

def correlated(df,threshold, drop_columns = False, encode_type='dumy'):
    '''Create a copy if you're viewing before deleting.
    If deleting df= correlated(df,...)'''


    if bool((df.select_dtypes('object')).size > 0):
        df = encode(df,encode_type)
        df_corr = df.corr()


    else:

        df_corr= df.corr()


    triangle = df_corr.where(np.triu(np.ones(df_corr.shape), k = 1).astype(bool))
    to_drop = pd.Series(df_corr.iloc[:,np.where((df_corr.mask(np.tril(np.ones(df_corr.shape,dtype=bool))).abs() > threshold).any())[0]].columns)

    if drop_columns:
        df.drop(labels=to_drop,axis=1,inplace=True)
        return df
        # print("Columns dropped:")
        # print(to_drop)
        # print(df.shape)





    else:

        collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])


        for i in to_drop:

            # Find the correlated features
            corr_features = list(triangle.index[triangle[i].abs() > threshold])

            # Find the correlated values
            corr_values = list(triangle[i][triangle[i].abs() > threshold])
            drop_features = [i for _ in range(len(corr_features))]

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                                'corr_feature': corr_features,
                                                'corr_value': corr_values})
            # Add to dataframe
            collinear = collinear.append(temp_df, ignore_index = True)



        else:
            print(f"From {len(df.columns)} columns")
            print(f"{len(to_drop)} highly correlated columns to drop:")
            print()
            print(to_drop)
            print("-----")
            print("Note_to_self...drop_feature may be duplicated due to multiple, stronger than threshold, correlated pairs:")
            print()
            print(collinear)

def identify_missing(df, missing_threshold, drop=False):
        """Find the features with a fraction of missing values above `missing_threshold`"""


        # Calculate the fraction of missing in each column
        missing_series = df.isnull().sum() / df.shape[0]
        missing_stats = pd.DataFrame(missing_series).rename(columns = {'index': 'feature', 0: 'missing_fraction'})

        # Sort with highest number of missing values on top
        missing_stats = missing_stats.sort_values('missing_fraction', ascending = False)

        # Find the columns with a missing percentage above the threshold
        record_missing = pd.DataFrame(missing_series[missing_series > missing_threshold]).reset_index().rename(columns =
                                                                                                               {'index': 'feature',
                                                                                                                0: 'missing_fraction'})

        to_drop = list(record_missing['feature'])

        if drop:
            df.drop(to_drop,axis=1,inplace=True)
            print(f"Columns dropped : {to_drop}")
            print(df.shape)

        else:
            print('%d features with greater than %0.2f missing values.\n' % (len(to_drop), missing_threshold))
            print(to_drop)

# FE

full_df = load_this_shit()
full_df.drop('Id',axis=1,inplace=True)
identify_missing(full_df,0.8)
identify_missing(full_df,0.8,True)
full_df = handle_missing(full_df)
#full_df.columns[full_df.isnull().any()]

skew_features = full_df[full_df.select_dtypes(['int64','float64']).columns].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index
print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
for i in skew_index:
    full_df[i] = boxcox1p(full_df[i], boxcox_normmax(full_df[i] + 1))

correlated(full_df, 0.8)
full_df = correlated(full_df, 0.8,True)
full_df.shape

# Pipeline Example
lightgbm = LGBMRegressor(objective='regression', num_leaves=6, learning_rate=0.01,n_estimators=7000, max_bin=200, bagging_fraction=0.8,bagging_freq=4, bagging_seed=8, feature_fraction=0.2, feature_fraction_seed=8, min_sum_hessian_in_leaf = 11, verbose=-1, random_state=42)

cont_features = full_df.select_dtypes(np.number).columns
cat_features = full_df.select_dtypes('object').columns

column_transformer = ColumnTransformer([
    (
        'numerical', 
        SimpleImputer(strategy='median'), 
        cont_features, 
    ), 
    (
        'categorical', 
        OneHotEncoder(handle_unknown='ignore'), 
        cat_features, 
    )
])
scaler = StandardScaler(with_mean=False)

preprocessing = Pipeline([
    ('column', column_transformer), 
    ('scaler', scaler), 
])

model = TransformedTargetRegressor(
    Pipeline(
        [
            ('preprocessing', preprocessing), 
            ('regressor', lightgbm
                
            ), 
        ], 
        verbose=False
    ), 
    transformer=QuantileTransformer()
)
model.fit(X_train, y_train)

# CV

bins = 5
sale_price_bins = pd.qcut(y, q=bins, labels=list(range(bins)))

X_train, X_test, y_train, y_test = train_test_split(
    full_df.iloc[:1460],
    y,
    random_state=12,
    stratify=sale_price_bins

)





# From book: Approaching almost any machine learning problem pg.110.

def create_folds(data):
    ''' For regression target needs to be binned in order to create groups which '''
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1
    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)
    # calculate the number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    num_bins = np.floor(1 + np.log2(len(data)))
    # bin targets
    data.loc[:, "bins"] = pd.cut(
    data["SalePrice"], bins=num_bins, labels=False
    )
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
        # drop the bins column
    data = data.drop("bins", axis=1)
        # return dataframe with folds
    return data
create_folds(train)

##########

def val_model(model, test): 
    # Validation Predictions
    test_predict = model.predict(X_test)
    # Viz Results
    plt.figure(figsize=(10, 7))
    plt.scatter(
        y_test, 
        test_predict, 
        marker='+', 
        label='Prediction'
    )
    plt.plot(
        [y_test.min(), y_test.max()], 
        [y_test.min(), y_test.max()],
        c='r',
        label='Perfect Prediction', 
    )
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.legend()
    plt.show()
    # Metric
    mse = mean_squared_error(y_test, test_predict)
    print(f'MSE: {mse}\tRMSE: {np.sqrt(mse)}')

val_model(model, y_test)   

#=======================
#PLOTTING

from plotnine import theme

ggplot(train , aes(train.SalePrice)) + geom_density() + theme_dark() 

plotnine.themes.theme_dark()

y.values
train = pd.read_csv('/home/r/Downloads/train.csv')


from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap
from plotnine.data import mtcars

(ggplot(mtcars, aes('wt', 'mpg', color='factor(gear)'))
 + geom_point()
 + stat_smooth(method='lm')
 + facet_wrap('~gear'))

#eda
import dtale
dtale.show(train)

prediction = model.predict(X_test)
np.sqrt(mean_squared_error(y_test, prediction))

#####################################################
#Imputation 

import miceforest
https://github.com/AnotherSamWilson/miceforest #The-MICE-Algorithm



from missingpy import MissForest
imputer = MissForest()
X_imputed = imputer.fit_transform(X)
https://github.com/epsilon-machine/missingpy


#================================

train = pd.read_csv('/home/r/Downloads/train.csv')
test = pd.read_csv('/home/r/Downloads/test.csv')


from sklearn import model_selection
df= train

# we create a new column called kfold and fill it with -1
df["kfold"] = -1
# the next step is to randomize the rows of the data
df = df.sample(frac=1).reset_index(drop=True)
# initiate the kfold class from model_selection module
kf = model_selection.KFold(n_splits=5)
# fill the new kfold column
for fold, (trn_, val_) in enumerate(kf.split(X=df)):
    df.loc[val_, 'kfold'] = fold

df.iloc[:,-1]

%timeit v
folds = KFold(n_splits = n_folds) # use TimeSeriesSplit cv
splits = folds.split(x_train, y_train)

#LGB
from time import time
import datetime
from sklearn import metrics

params = {
    'boosting_type': 'gbdt',
    'max_depth': -1,
    'objective': 'regression',
    'is_unbalance': False,
    'nthread': 3,
    # 'num_leaves': 41,
    'learning_rate': 0.05,
    'max_bin': 50,
    'subsample_for_bin': 200,
    'subsample': 1,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 5,
    'reg_lambda': 10,
    # 'min_split_gain': 0.5,
    'min_child_weight': 1,
    'min_child_samples': 5,
    'scale_pos_weight': 1,
    # 'min_sum_hessian_in_leaf': 226,
    'num_class': 1,
    'metric': 'mse',
    'verbose': -1,
    'bagging_fraction': 0.5540557845037332,
    'bagging_freq': 1,
    'feature_fraction': 0.5068213222676954,
    'lambda_l1': 1,
    'lambda_l2': 0.6088814046649453,
    'max_depth': 7,
    'min_data_in_leaf': 239,
    'min_gain_to_split': 1,
    'min_sum_hessian_in_leaf': 225,
    'num_leaves': 45,
    'weight': 234.38737874771672}

def lgb_regression(x_train, x_test, y_train,params, n_folds=5,plot_feature_importance=False, verbose=100, early_stopping_rounds=100,num_boost_round=100):

    training_start_time = time()

    features = list(x_train.columns)
    oof = np.zeros(len(X_train))
    predictions = np.zeros(len(y_train))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(x_train,y_train)):

        print(f'Training fold {fold + 1}')

        train_set= lgb.Dataset(x_train.iloc[trn_idx], label=y_train.iloc[trn_idx])
        val_set = lgb.Dataset(x_train.iloc[val_idx], label=y_train.iloc[val_idx])

        lgb_model = lgb.train(params,
                          train_set= train_set,
                          valid_sets= [train_set, val_set],
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping_rounds,
                          )

        oof[val_idx] = lgb_model.predict(x_train.iloc[val_idx][features],
                                    num_iteration = lgb_model.best_iteration)

        #predictions += lgb_model.predict(x_test[features])/n_folds
        print('-'*50)
        print('\n')

    oof_rmse = np.sqrt(metrics.mean_squared_error(y_train, oof))
    print(f"Total training time: {str(datetime.timedelta(seconds=time() - training_start_time)).split('.')[0]}")
    print(f'RMSE: is {oof_rmse}')
    return lgb_model

lgb_regression(x_train=X_train, x_test=X_test, y_train=y_train,params=params,verbose=200)

lgb_model



if: plot_feature_importance=False

lgb.plot_importance(model, importance_type = 'gain', precision = 0,
                            height = 0.5, figsize = (6, 10),
                            title = f'fold {fold} feature importance',
                            ignore_zero = True)



skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
import  time
time.time()
from sklearn import model_selection
kf = model_selection.KFold(n_splits=5)
fold = model_selection.KFold(n_splits=5)
for fold, (trn_idx, test_idx) in enumerate(kf.split(X, y)):
    start_time = time()
    print('Training on fold {}'.format(fold + 1))

    trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])
    clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=verbose, early_stopping_rounds=500,)

    feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()
    aucs.append(clf.best_score['valid_1']['auc'])

    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
print('-' * 30)
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print('Mean AUC:', np.mean(aucs))
print('-' * 30)


##############################################


from sklearn import metrics

def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()
    

def train_model_regression(X, X_test, y, params, folds, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200, n_estimators=50000):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type

    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]
    
    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                        'catboost_metric_name': 'MAE',
                        'sklearn_scoring_function': metrics.mean_absolute_error},
                    'group_mae': {'lgb_metric_name': 'mae',
                        'catboost_metric_name': 'MAE',
                        'scoring_function': group_mean_log_mae},
                    'mse': {'lgb_metric_name': 'mse',
                        'catboost_metric_name': 'MSE',
                        'sklearn_scoring_function': metrics.mean_squared_error}
                    }

    
    result_dict = {}
    
    # out-of-fold predictions on train data
    oof = np.zeros(len(X))
    
    # averaged predictions on train data
    prediction = np.zeros(len(X_test))
    
    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()
    
    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = n_estimators, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')
            
            y_pred = model.predict(X_test).reshape(-1,)
        
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))

        prediction += y_pred    
        
        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= folds.n_splits
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    
    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= folds.n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
            
            result_dict['feature_importance'] = feature_importance
        
    return result_dict

#%%
import time
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

result_dict = train_model_regression(X_train,X_test,y,params,fold,'lgb','mse',plot_feature_importance=True)
result_dict
#%%
params = {
        'boosting_type': 'gbdt',
        
        'seed': 42,
        'learning_rate': 0.1,
        'bagging_fraction': 0.85,
        'bagging_freq': 1, 
        'colsample_bytree': 0.85,
        'colsample_bynode': 0.85,
        'min_data_per_leaf': 25,
        'num_leaves': 200,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5}


######################ANDRADA
def training_lgbm(train_lgbm, test_lgbm, features, target, param,
                  n_splits=5, stop_round=100, num_rounds=1000, verbose=False,
                  tuned="None", val=None, return_model=False, step=1):
    '''Trains LGBM model.'''
    run = wandb.init(project='wids-datathon-kaggle', name=f'lgbm_run_{step}',
                     config=param)
    wandb.log(param)

    # ~~~~~~~
    #  KFOLD
    # ~~~~~~~
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    oof = np.zeros(len(train_lgbm))
    predictions = np.zeros(len(test_lgbm))

    # Convert Train to Train & Validation
    skf_split = skf.split(X=train_lgbm[features], y=train_lgbm[target].values)

    # ~~~~~~~
    #  TRAIN
    # ~~~~~~~
    counter = 1

    for train_index, valid_index in skf_split:
        print("==== Fold {} ====".format(counter))

        lgbm_train = lgbm.Dataset(data=train_lgbm.iloc[train_index, :][features].values,
                                  label=train_lgbm.iloc[train_index, :][target].values,
                                  feature_name=features,
                                  free_raw_data=False)

        lgbm_valid = lgbm.Dataset(data=train_lgbm.iloc[valid_index, :][features].values,
                                  label=train_lgbm.iloc[valid_index, :][target].values,
                                  feature_name=features,
                                  free_raw_data=False)

        lgbm_1 = lgbm.train(params=param, train_set=lgbm_train, valid_sets=[lgbm_valid],
                            early_stopping_rounds=stop_round, num_boost_round=num_rounds,
                            verbose_eval=verbose, callbacks=[wandb_callback()])

        # X_valid to predict
        oof[valid_index] = lgbm_1.predict(train_lgbm.iloc[valid_index][features].values,
                                          num_iteration=lgbm_1.best_iteration)
        predictions += lgbm_1.predict(test_lgbm[features],
                                      num_iteration=lgbm_1.best_iteration) / n_splits

        counter += 1

    # ~~~~~~~~~~~
    #   OOF EVAL
    # ~~~~~~~~~~~
    print("============================================")
    print("Splits: {} | Stop Round: {} | No. Rounds: {} | {}: {}".format(n_splits, stop_round,
                                                                         num_rounds, tuned, val))
    print("CV ROC: {:<0.5f}".format(metrics.roc_auc_score(test_lgbm[target], predictions)))
    print("\n")
    wandb.log({'oof_roc': metrics.roc_auc_score(test_lgbm[target], predictions)})
    wandb.finish()

    if return_model:
        return lgbm_1

######################


train = pd.read_csv('/home/r/Downloads/train.csv')
train

cat = [i for i in train.columns if 'cat' in i]

