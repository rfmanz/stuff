import copy
import pandas as pd
import optuna
import lightgbm as lgb
from rdsutils.metrics import score_gain

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold



def ks_score(y_true, y_pred, sample_weight=None, 
             nr_quantiles=100, round_by=4):
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred)
    ks = score_gain(y_true, y_pred, sample_weight=sample_weight,
                    nr_quantiles=nr_quantiles, direction=1,
                    round_by=round_by)['KS'].max()
    return ks


def get_trial_history(study):
    from optuna.trial import TrialState
    results = [trial.values for trial in study.trials if trial.state==TrialState.COMPLETE]
    results = [r[0] for r in results if isinstance(r, list)]
    return results


def objective_builder(train_df, valid_df, features, target_col,
                      base_params, weight_col=None, eval_weight_col=None,
                      metric="auc"):
    """
    function that returns a function
    """
    # data processing
    wtr, wts = None, None
    if weight_col:
        wtr = train_df[weight_col]
        if weight_col in valid_df.columns:
            wts = valid_df[weight_col]
    if eval_weight_col:
        wts = valid_df[eval_weight_col]
        
    print("full df size: ", train_df.shape, valid_df.shape)
    print("feature size: ", train_df[features].shape, valid_df[features].shape)
    
    # build datasets once
    dtrain = lgb.Dataset(train_df[features], 
                         label=train_df[target_col], 
                         weight=wtr)
    dvalid = lgb.Dataset(valid_df[features], 
                         label=valid_df[target_col],
                         weight=wts)
        
    def objective(trial):
        
        params = copy.deepcopy(base_params)
        trial_params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 3000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8), 
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 50.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 50.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 512, step=4),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
#             "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 1000, step=200)
        }
        params.update(trial_params)
#         print(params)
        
        # Add a callback for pruning.
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, params["metric"])
        clf = lgb.train(
            params, dtrain, valid_sets=[dvalid], verbose_eval=False, callbacks=[pruning_callback]
        )
        preds = clf.predict(valid_df[features])
        
        if metric == "auc":
            m = roc_auc_score(valid_df[target_col], preds, sample_weight=valid_df[eval_weight_col])
        elif metric == "ks":
            m = ks_score(valid_df[target_col], preds, sample_weight=valid_df[eval_weight_col])
        return m
    
    return objective




def cv_objective_builder(df, target_col, features, 
                         weight_col=None, eval_weight_col=None, metric="auc", 
                         num_folds=4):
    """
    cv_objective = cv_objective_builder(df, target, features, weight, weight)
    """

    def cv_objective(trial):
        kf = KFold(num_folds)
        split = kf.split(X=df, y=df[target_col])

        pred = np.zeros(len(df))
        
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "early_stopping_rounds": 50,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 512),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        
        for i, (train_idx, test_idx) in tqdm(enumerate(split)):
            train = df.iloc[train_idx].copy()
            test = df.iloc[test_idx].copy()

            wtr = None
            wts = None
            if weight_col:
                wtr = train[weight_col]
                if weight_col in test:
                    wts = test[weight_col]
            if eval_weight_col:
                wts = test[eval_weight_col]
            
            dtrain = lgb.Dataset(train[features], label=train[target_col], weight=wtr)
            dvalid = lgb.Dataset(test[features], label=test[target_col], weight=wts)

            # Add a callback for pruning.
            pruning_callback = optuna.integration.LightGBMPruningCallback(trial, metric)
            clf = lgb.train(
                params, dtrain, valid_sets=[dvalid], verbose_eval=False, callbacks=[pruning_callback]
            )

            pred[test_idx] = clf.predict(test[features],
                                         num_iteration=clf.best_iteration)
        
        if metric == "auc":
            return roc_auc_score(df[target_col], pred, sample_weight=df[eval_weight_col])
        return None
    
    return cv_objective


