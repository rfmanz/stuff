import lightgbm as lgb
import pandas as pd
import copy
import gc
from sklearn.model_selection import train_test_split, \
    GroupKFold, StratifiedKFold


def get_default_lgb_estimator_(task, n_estimators=1000,
                              class_weight=None, **kwargs):
    """
    @params task: What type of task: ['classification', 'refression']
    @params n_estimators: number of estimators to have in lgb
    @params class_weight: weights for each class
    @params **kwargs: all other features lgb takes
    
    @returns model: a lightgbm model
    """
    if task == 'classification':
        model = lgb.LGBMClassifier(n_estimators=n_estimators,
                                   class_weight=class_weight,
                                   **kwargs)
    elif task == 'regression':
        model = lgb.LGBMRegressor(n_estimators=n_estimators,
                                  class_weight=class_weight,
                                  **kwargs)
    else:
        raise NotImplemented('Please provide a valid task type :\
        ["classification", "regression"]')
    return model


def get_lgb_importance_(df, features, task, label_cols, model=None, eval_metric=None,
                       n_iterations=5, early_stopping_rounds=100, 
                       cat_features=None, n_estimators=1000,
                       class_weight=None, group_col_name=None, **kwargs):
    """ Get LightGBM importance 
    @params df: dataframe
    @params features: features the lgb will take
    @params task: What type of task: ['classification', 'refression']
    @params eval_metric: lightgbm eval metrics, e.g. auc, logloss, mse
    @params n_iterations: number of iterations to run
    @params early_stopping_rounds: num rounds to early stop
    @params cat_features: list of categorical features
    @params group_col_name: id column used to do GroupKFold on
    
    @returns fimp_folds: feature importances for all folds
    @returns fimp_mean: mean feature importance
    """
    cols = copy.copy(features)
    if group_col_name is not None:
        cols = cols + group_col_name
    data = df[cols]
    labels = df[label_cols].values

    # int encode categorical variables
    df = categorical_encoder(data, cat_features)

    fimp_df = pd.DataFrame()
    if group_col_name is None:
        print('stratifying by labels')
        kf = StratifiedKFold(n_splits=n_iterations)
        split = kf.split(X=df, y=labels)
    else:
        print('GroupKFold by group_col_name')
        kf = GroupKFold(n_splits=n_iterations)
        groups = df[group_col_name]
        split = kf.split(X=df, y=labels, groups=groups)

    for i, (trn_idx, val_idx) in enumerate(split):
        print(f'fitting models! iteration {i}\r')
        if model is None:
            model_ = get_default_lgb_estimator_(task,
                                                n_estimators=n_estimators,
                                                class_weight=class_weight,
                                                **kwargs)
        else:
            model_ = copy.deepcopy(model)

        if early_stopping_rounds:
            trn_x, trn_y = df.iloc[trn_idx], labels[trn_idx]
            val_x, val_y = df.iloc[val_idx], labels[val_idx]
            model_.fit(trn_x[features], trn_y, 
                       eval_set=[(val_x[features], val_y)], 
                       early_stopping_rounds=100,
                       verbose=-1)

            del trn_x, val_x, trn_y, val_y
            gc.collect()

        else:
            model_.fit(df, labels)

        fimp_df_ = pd.DataFrame()
        fimp_df_['feature'] = features
        fimp_ = model_.feature_importances_
        fimp_df_['importance'] = fimp_ / fimp_.sum() * 100 # normalize importance
        fimp_df_['fold'] = i + 1
        fimp_df = pd.concat([fimp_df, fimp_df_], axis=0)

    fimp_folds = fimp_df
    fimp_mean = fimp_df.groupby('feature')['importance'].mean()
    fimp_mean.sort_values(ascending=False, inplace=True)
    fimp_mean = fimp_mean.rename('importance').to_frame().reset_index()
    return fimp_folds, fimp_mean
 
    
def categorical_encoder(df, cols):
    for col in cols:
        cats = df[col].unique()
        map_dict = dict(zip(cats, range(len(cats))))
        df[col] = df[col].map(map_dict)
    return df