import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

import optuna
import optuna.integration.lightgbm as lgb
from optuna.pruners import SuccessiveHalvingPruner

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import joblib
import os




def optuna_LGBMClassifier_tuner(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test, n_trials=30,params = None):
  """paramsLGBM = optuna_LGBMClassifier_tuner(...)"""

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
          'device': 'cpu',
          'objectve' : 'binary'
      }
  
      model = LGBMClassifier(**params)
      model.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=222, verbose=False)
      y_pred = model.predict_proba(x_test)[:, 1]
      roc_auc = roc_auc_score(y_test, y_pred)
      return roc_auc

  study = optuna.1create_study(direction='maximize',pruner=SuccessiveHalvingPruner())
  study.optimize(objective, n_trials=n_trials)
  print('Number of finished trials:', len(study.trials))
  print('Best trial:', study.best_trial.params)
  print('Best value:', study.best_value)
  paramsLGBM = study.best_trial.params
  return paramsLGBM


def model_LGBMClassifier(x=x_train, y=y_train, test=x_test, y_test=y_test, Kfold_splits = 10, paramsLGBM=paramsLGBM , early_stopping_rounds =500):

  kf = KFold(n_splits=Kfold_splits, shuffle=True, random_state=42)
  auc = []
  preds = np.zeros(test.shape[0])
  
  
  for fold, (trn_idx, val_idx) in enumerate(kf.split(x, y)):
      print(f"===== FOLD {fold+1} =====")
      x_train, x_val = x.iloc[trn_idx], x.iloc[val_idx]
      y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
  
      model = LGBMClassifier(**paramsLGBM)
  
      model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='auc', verbose=False,early_stopping_rounds=500)
  
      auc.append(roc_auc_score(y_val, model.predict_proba(x_val)[:, 1]))
      
      preds += model.predict_proba(test)[:, 1] / kf.n_splits
  
  feature_importances_extra = pd.Series(model.feature_importances_,x_train.columns).sort_values(ascending=False)
  
  print()
  print("-"*80)
  print("CV AUC MEAN:",np.mean(auc))
  print()
  print("-"*80)
  print("CV Feature Importance:")
  print("-"*80)
  print(feature_importances_extra)
  print("-"*80)
  print("MODEL AUC PERFORMANCE ON TEST:", roc_auc_score(y_test,preds))
  print("-"*80)
  return model


def save_model(model = model, name = "something.sav"):
  joblib.dump(model, name)
  print("Model_Name:",name)
  print("Saved to:", os.getcwd())
    

def load_model(name="somethng.sav"):
  model=  joblib.load(name)
  return model 



































# from time import time
# import datetime
# from sklearn import metrics
# from sklearn.model_selection import StratifiedKFold
# import lightgbm as lgb
#
# # define random hyperparammeters
# # params = {
# #     'boosting_type': 'gbdt',
# #     'n_jobs': -1,
# #     'seed': 42,
# #     'learning_rate': 0.1,
# #     'bagging_fraction': 0.85,
# #     'bagging_freq': 1,
# #     'colsample_bytree': 0.85,
# #     'colsample_bynode': 0.85,
# #     'min_data_per_leaf': 25,
# #     'num_leaves': 200,
# #     'lambda_l1': 0.5,
# #     'lambda_l2': 0.5}
#
#
# params = {'boosting_type': 'gbdt',
#           'objective': "regression",
#           'metric': 'rmse',
#           'max_bin': 300,
#           'max_depth': 5,
#           'num_leaves': 200,
#           'learning_rate': 0.01,
#           'feature_fraction': 0.7,
#           'bagging_fraction': 0.7,
#           'bagging_freq': 10,
#           'verbose': 0,
#           'num_threads': 1,
#           'lambda_l2': 3,
#           'min_gain_to_split': 0,
#           }
#
#
# def train_lgbm(x_train, x_test, y_train, params, n_folds=5, early_stopping_rounds=100, num_boost_round=100):
#     training_start_time = time()
#
#     features = list(x_train.columns)
#     oof = np.zeros(len(x_train))
#     predictions = np.zeros(len(y_train))
#     skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
#
#     for fold, (trn_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
#         print(f'Training fold {fold + 1}')
#
#         train_set = lgb.Dataset(x_train.iloc[trn_idx], label=y_train.iloc[trn_idx])
#         val_set = lgb.Dataset(x_train.iloc[val_idx], label=y_train.iloc[val_idx])
#
#         lgb_model = lgb.train(params,
#                               train_set=train_set,
#                               valid_sets=[train_set, val_set],
#                               num_boost_round=num_boost_round,
#                               early_stopping_rounds=early_stopping_rounds)
#
#         oof[val_idx] = lgb_model.predict(x_train.iloc[val_idx][features],
#                                          num_iteration=lgb_model.best_iteration)
#
#         # predictions += lgb_model.predict(x_test[features])/n_folds
#         print('-' * 50)
#         print('\n')
#
#     oof_rmse = np.sqrt(metrics.mean_squared_error(y_train, oof))
#     print(f"Total training time: {str(datetime.timedelta(seconds=time() - training_start_time)).split('.')[0]}")
#     print(f'RMSE: is {oof_rmse}')
#     return lgb_model
#
#
