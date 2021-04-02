from time import time
import datetime
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

# define random hyperparammeters
# params = {
#     'boosting_type': 'gbdt',
#     'n_jobs': -1,
#     'seed': 42,
#     'learning_rate': 0.1,
#     'bagging_fraction': 0.85,
#     'bagging_freq': 1,
#     'colsample_bytree': 0.85,
#     'colsample_bynode': 0.85,
#     'min_data_per_leaf': 25,
#     'num_leaves': 200,
#     'lambda_l1': 0.5,
#     'lambda_l2': 0.5}


params = {'boosting_type': 'gbdt',
          'objective': "regression",
          'metric': 'rmse',
          'max_bin': 300,
          'max_depth': 5,
          'num_leaves': 200,
          'learning_rate': 0.01,
          'feature_fraction': 0.7,
          'bagging_fraction': 0.7,
          'bagging_freq': 10,
          'verbose': 0,
          'num_threads': 1,
          'lambda_l2': 3,
          'min_gain_to_split': 0,
          }


def train_lgbm(x_train, x_test, y_train, params, n_folds=5, early_stopping_rounds=100, num_boost_round=100):
    training_start_time = time()

    features = list(x_train.columns)
    oof = np.zeros(len(X_train))
    predictions = np.zeros(len(y_train))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        print(f'Training fold {fold + 1}')

        train_set = lgb.Dataset(x_train.iloc[trn_idx], label=y_train.iloc[trn_idx])
        val_set = lgb.Dataset(x_train.iloc[val_idx], label=y_train.iloc[val_idx])

        lgb_model = lgb.train(params,
                              train_set=train_set,
                              valid_sets=[train_set, val_set],
                              num_boost_round=num_boost_round,
                              early_stopping_rounds=early_stopping_rounds)

        oof[val_idx] = lgb_model.predict(x_train.iloc[val_idx][features],
                                         num_iteration=lgb_model.best_iteration)

        # predictions += lgb_model.predict(x_test[features])/n_folds
        print('-' * 50)
        print('\n')

    oof_rmse = np.sqrt(metrics.mean_squared_error(y_train, oof))
    print(f"Total training time: {str(datetime.timedelta(seconds=time() - training_start_time)).split('.')[0]}")
    print(f'RMSE: is {oof_rmse}')
    return lgb_model
