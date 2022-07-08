import json, os, sys, io, boto3
import pandas as pd


# with open("../../config.json","r") as f:
#     config = json.load(f)
    
# SEED = config["model_params"]["seed"]
# TARGET_COL = "target"
# INDETERMINATE_COL = config["indeterminate_col"]


def scale_score(df, pred_col):
    # raise NotImplemented
    assert(df[pred_col].between(0,1).all())  # assert pred is probability
    return df[pred_col] * 100
    

def task_specific_data_preprocessing(config, modeling_df, valid_dfs=None, 
                                     test_dfs=None, target_col="target"):
    # raise NotImplemented
    if target_col not in modeling_df.columns:  # rename target column if needed
        modeling_df[target_col] = modeling_df[config["target_col"]]
        
        modeling_df['account_ending_balance'] = modeling_df['real_ending_balance']
        modeling_df['days_since_first_transaction'] = modeling_df['days_since_first_deposit']

        if valid_dfs is not None:
            for fname, valid_df in valid_dfs.items(): 
                valid_df[target_col] = valid_df[config["target_col"]]
                valid_df['account_ending_balance'] = valid_df['real_ending_balance']
                valid_df['days_since_first_transaction'] = valid_df['days_since_first_deposit']

        if test_dfs is not None:
            for fname, test_df in test_dfs.items():
                test_df['account_ending_balance'] = test_df['real_ending_balance']
                test_df['days_since_first_transaction'] = test_df['days_since_first_deposit']
            
    return modeling_df, valid_dfs, test_dfs
        
    
def prep_data(config, target_col="target"):
    
    # load data
    path = config["modeling_data_path"]
    print(f"loading modeling_df from: {path}")
    modeling_df = pd.read_parquet(path)

    valid_dfs = {}
    for fname, path in config["validation_data_paths"].items():
        print(f"loading valid_df from: {path}")
        valid_dfs[fname] = pd.read_parquet(path)
        
    test_dfs = {}
    for fname, path in config["inference_data_paths"].items():
        print(f"loading test_df from: {path}")
        test_dfs[fname] = pd.read_parquet(path)

    # additional processing
    modeling_df, valid_dfs, test_dfs = task_specific_data_preprocessing(config, modeling_df, 
                                                                        valid_dfs, test_dfs,
                                                                        target_col=target_col)
    
    return modeling_df, valid_dfs, test_dfs


def get_pickle_from_s3(bucket, obj_path):
    import boto3, io, pickle as pkl
    s3 = boto3.client('s3')
    pickle_buffer = io.BytesIO()

    model = pkl.loads(s3.get_object(Bucket=bucket, Key=obj_path)["Body"].read())
    return model


def get_baseline_models(config):
    # raise NotImplemented
    # !pip install "scikit-learn==0.22" for deposit_v1_model

    benchmarks = {}
    for mname, s3_kwargs in config["baseline_models"].items():
        benchmarks[mname] = {"model": get_pickle_from_s3(s3_kwargs["bucket"], s3_kwargs["key"])}
    return benchmarks


def get_baseline_preds(models, modeling_df, valid_dfs, test_dfs):
    """
    models have structure:
    {"name1":{"model":model_obj, "preprocess": python_function},
     "name2": ...}
    """
    import gc
    
    new_cols = []
    
    for mname, obj in models.items():
        model = obj["model"]
        preprocess = obj["preprocess"]
              
        features = model.feature_name_
        best_iter = model.best_iteration_
        
        # temp fix
        if mname == "deposit_v1":
            def preprocess(df):
                for feature in model.feature_name_:
                    df[feature] = df[feature].astype(float)
                return df
        
        # eval on the dataframes
        # raise warning: if column already exist in df
        
        df_ = preprocess(modeling_df.copy())
        new_cols.append(f"{mname}_pred")
        modeling_df[f"{mname}_pred"] = model.predict_proba(df_[features],
                                                           num_iteration=best_iter)[:,1]
        
        print(f"generating {mname} prediction for modeling_df")
        
        del df_
        gc.collect()
        
        for fname, valid_df in valid_dfs.items():
            df_ = preprocess(valid_df.copy())
            valid_df[f"{mname}_pred"] = model.predict_proba(df_[features],
                                                            num_iteration=best_iter)[:,1]
            
            print(f"generating {mname} prediction for {fname}")
            del df_
            gc.collect()
            
        for fname, test_df in test_dfs.items():
            df_ = preprocess(test_df.copy())
            test_df[f"{mname}_pred"] = model.predict_proba(df_[features], 
                                                           num_iteration=best_iter)[:,1]
            
            print(f"generating {mname} prediction for {fname}")
            del df_
            gc.collect()
        
    return modeling_df, valid_dfs, test_dfs, new_cols



def get_model_preds(model, modeling_df, valid_dfs, test_dfs, preprocess=None):
    """
    models have structure:
    {"name1":{"model":model_obj, "preprocess": python_function},
     "name2": ...}
    """
    import gc
    
    
    mname = "final model"
    features = model.feature_name_
    best_iter = model.best_iteration_
    new_cols = ["pred", "score"]
    
    if preprocess is None:
        preprocess = lambda df: df
    # eval on the dataframes
    # raise warning: if column already exist in df

    df_ = preprocess(modeling_df.copy())
    modeling_df[f"pred"] = model.predict_proba(df_[features],
                                                       num_iteration=best_iter)[:,1]
    modeling_df[f"score"] = scale_score(modeling_df, f"pred")
    print(f"generating {mname} prediction for modeling_df")

    del df_
    gc.collect()

    for fname, valid_df in valid_dfs.items():
        df_ = preprocess(valid_df.copy())
        valid_df[f"pred"] = model.predict_proba(df_[features],
                                                        num_iteration=best_iter)[:,1]
        valid_df[f"score"] = scale_score(valid_df, f"pred")
        print(f"generating {mname} prediction for {fname}")
        del df_
        gc.collect()

    for fname, test_df in test_dfs.items():
        df_ = preprocess(test_df.copy())
        test_df[f"pred"] = model.predict_proba(df_[features], 
                                                       num_iteration=best_iter)[:,1]
        test_df[f"score"] = scale_score(test_df, f"pred")
        print(f"generating {mname} prediction for {fname}")
        del df_
        gc.collect()
        
    return modeling_df, valid_dfs, test_dfs, new_cols


def train_model(config, context, preprocess_fn):
    import lightgbm as lgb 

    params = config["model_params"].copy()
    params["early_stopping_round"] = None
    features = config["features"]
    modeling_df = preprocess_fn(context["modeling_df"].copy())
    print("preprocess modeling_df for training")
    indet_col = context["indeterminate_col"]
    
    if indet_col in modeling_df.columns:
        modeling_df = modeling_df[~modeling_df[indet_col]]

    clf = lgb.LGBMClassifier(**params)
    X = modeling_df[features]
    y = modeling_df[context["target_col"]]
    clf.fit(X, y)
    
    context["model_object"] = clf
    return config, context
    
    
def validate_model(config, context, preprocess_fn):

    modeling_df = context["modeling_df"]
    valid_dfs = context["valid_dfs"]
    test_dfs = context["test_dfs"]
    clf = context["model_object"]
    
    result = get_model_preds(clf, modeling_df, valid_dfs, test_dfs, preprocess_fn)
    modeling_df, valid_dfs, test_dfs, new_cols = result
    context["pred_cols"].extend(new_cols)
    
    context["modeling_df"] = modeling_df
    context["valid_dfs"] = valid_dfs
    context["test_dfs"] = test_dfs
    return config, context