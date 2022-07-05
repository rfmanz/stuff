import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def train_lgb_baseline_grouped(df, features, params, target_col_name='target', 
                               group_col_name='business_account_number', num_folds=5, seed=12345, 
                               n_boost_rounds=100000, early_stopping_rounds=1000, 
                               verbose_eval=500, weight_col_name=None, cat_features=None, prepro=None):
    """
    Train LightGBM models over 5 folds, return OOF predictions, trained models, and average AUC over 5 folds.
    """
    kf = GroupKFold(num_folds)
    split = kf.split(X=df, y=df[target_col_name], groups=df[group_col_name])

    train_pred = np.zeros(len(df))
    feature_importance_df = pd.DataFrame()

    models = []

    for i, (train_idx, test_idx) in enumerate(split):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        if weight_col_name:
            wtr = train[weight_col_name]
            wts = test[weight_col_name]
        else:
            wtr = None
            wts = None
            
        if prepro is not None:
            preprocessor = prepro
            train[cat_features] = prepro.fit_transform(X=train[cat_features], y=train[target_col_name])
            test[cat_features] = prepro.transform(test[cat_features])

        lgb_train = lgb.Dataset(train[features],
                                label=train[target_col_name],
                                free_raw_data=False,
                                weight=wtr)
        lgb_test = lgb.Dataset(test[features],
                               label=test[target_col_name],
                               free_raw_data=False,
                               weight=wts)

        model = lgb.train(params,
                          lgb_train,
                          valid_sets=[lgb_train, lgb_test],
                          valid_names=['train', 'test'],
                          num_boost_round=n_boost_rounds,
                          early_stopping_rounds= early_stopping_rounds,
                          verbose_eval=verbose_eval)
        models.append(model)

        train_pred[test_idx] = model.predict(test[features], num_iteration=model.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = model.feature_importance(importance_type='gain', 
                                                                    iteration=model.best_iteration)
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    auc = roc_auc_score(y_true=df[target_col_name].values, y_score=train_pred)

    return train_pred, feature_importance_df, models, auc


def train_validate_lgb(df_train, df_test, features, params,
                       target_col_name='target', seed=12345, 
                       n_boost_rounds=100000, early_stopping_rounds=500, 
                       verbose_eval=500, weight_col_name=None, 
                       cat_features=None, prepro=None):
    """
    Train a LightGBM model and return the OOF predictions
    Train on df_train and test on df_test
    
    prepro designed to take into categorical_encoders like:
        https://contrib.scikit-learn.org/category_encoders/index.html
        
    """

    feature_importance_df = pd.DataFrame()

    train = df_train.copy()
    test = df_test.copy()

    if weight_col_name:
        wtr = train[weight_col_name]
        wts = test[weight_col_name]
    else:
        wtr = None
        wts = None
    
    if prepro is not None:
        preprocessor = prepro
        train[cat_features] = prepro.fit_transform(X=train[cat_features],
                                                  y=train[target_col_name])
        test[cat_features] = prerpo.transform(test[cat_features])
    
    lgb_train = lgb.Dataset(train[features],
                            label=train[target_col_name],
                            free_raw_data=False,
                            weight=wtr)
    lgb_test = lgb.Dataset(test[features],
                           label=test[target_col_name],
                           free_raw_data=False,
                           weight=wts)
    
    model = lgb.train(params,
                      lgb_train,
                      valid_sets=[lgb_train, lgb_test],
                      valid_names=['train', 'test'],
                      num_boost_round=n_boost_rounds,
                      early_stopping_rounds= early_stopping_rounds,
                      verbose_eval=verbose_eval)

    test['pred'] = model.predict(test[features], num_iteration=model.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = model.feature_importance(importance_type='gain', 
                                                                iteration=model.best_iteration)
    fold_importance_df["fold"] = 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    auc = roc_auc_score(y_true=test[target_col_name].values, y_score=test.pred)
    
    return test.pred, feature_importance_df, model, auc
