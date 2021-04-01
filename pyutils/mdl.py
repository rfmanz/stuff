from time import time
import datetime
from sklearn import metrics



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