import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
import time
import random
import seaborn as sns

def lgbm_model_trainer(df, target, features, model,if_validate_df=None, model_name=None,
                       weight=None, indeterminate = None, train_ratio = 0.75):
    start_time = time.time()
    if indeterminate != None:
        print("data statistics", df.shape)
        len1 = len(df)
        print("indeterminate removed! column name: ", indeterminate)
        df[indeterminate] = df[indeterminate].astype('bool')
        df = df[~df[indeterminate]]
        len2 = len(df)
        print("data statistics after removing indeterminate", df.shape)
        print("indetermineate: ", 1-len2/len1)
        
    random.seed(12345)
    eval_set = None
    df_train=df.copy()
    if if_validate_df is not None:
        app_train = random.sample(df.id.unique().tolist(), 
                                  round(df.id.unique().size * train_ratio))
        df_train = df[df.id.isin(app_train)].copy()
        df_valid = df[~df.id.isin(app_train)].copy()
        eval_set=[(df_valid[features], df_valid[target])]     

        model.fit(df_train[features], df_train[target] ,sample_weight=df_train[weight],
                  eval_set=eval_set, eval_sample_weight = df_valid[weight], early_stopping_rounds=50, verbose = 100)
    else:
        model.fit(df_train[features], df_train[target] ,sample_weight=df_train[weight],
                 verbose = 100)        
    
    model_elapse = time.time() - start_time
    print('training elapse:, ', model_elapse/60, ' mins')
    
    sns.set(rc={'figure.figsize':(12,8)})
#    lgb.plot_metric(model, metric='auc', ylim=(0,1))
   
    if model_name!=None:
        joblib.dump(model, '../../models/'+ model_name +'.pkl')
    
    if if_validate_df is not None:
        return model, df_train, df_valid
    else:
        return model