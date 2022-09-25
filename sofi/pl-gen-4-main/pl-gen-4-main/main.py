import sys, os, json, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.logger import make_logger
from src.dataloader import TabularDataloader
from src.Trainer import LGBMTrainer, TFTrainer
from src.utils.metrics import get_pred_reports
from src.logger import make_logger
     

def main(config: dict, debug=None):
    # get config
    with open(os.path.join(config), "r") as f:
        config = json.load(f)
           
    mname = config["meta"]["model_name"]
    
    global log_er
    log_er, _ = make_logger(__name__,
                            f"./models/{mname}/log.log",
                            overwrite_file=True)
    
    log_er.info(f"config loaded - {mname}")
    
    # get dataloader
    train_path = config["data_processed"]["xf_train_df"]
    valid_path = config["data_processed"]["xf_valid_df"]
    test_path = config["data_processed"]["xf_test_df"]
    target_col = config["data_columns"]["target_cols"][0]
    
    dl = TabularDataloader(train_path, 
                       valid_path,
                       test_path,
                       target_col)
    dl.load_data(debug_size=debug)
    train_df, valid_df, test_df = dl.get_data(debug=False)

    log_er.info("data loaded")
    log_er.info(f"shapes: {str(train_df.shape)} - {str(valid_df.shape)} - {str(test_df.shape)}")
    
    # get Trainer
    if config["meta"]["model_type"] == "lightgbm":
        run_lgbm(config, train_df, valid_df, test_df)
    elif config["meta"]["model_type"] in ["mlp", "tabnet"]:
        run_tf(config, train_df, valid_df, test_df)
    else:
        raise NotImplementedError
        
    # if validate - produce valid result
    
    # if test - produce test result

    # auxiliary
    return None


def run_lgbm(config, train_df, valid_df, test_df):
    # getting setups
    mname = config["meta"]["model_name"]
    features = config["data_columns"]["gen3_features"]
    params = config["model_params"]
    target_col = config["data_columns"]["target_cols"][0]

    # setup model
    lgbm = lgb.LGBMClassifier(**params)
    
    # setup trainer
    trainer = LGBMTrainer()
    trainer.train(lgbm, 
                  train_df,
                  features = features,
                  target_col = target_col,
                  valid_df = valid_df,
                  early_stopping_rounds=5)
    log_er.info("model fitted")

    save_path = f"./models/{mname}/model.pkl"
    trainer.save_model(save_path)
    log_er.info("model saved at: " + save_path)

    pred_test = trainer.predict(test_df)[:,1]
    test_df["pred"] = pred_test

    metrics = get_pred_reports(test_df, target_col, ['pred'])
    metrics.to_csv(f"./models/{mname}/metrics.csv")
    log_er.info("metrics: " + str(metrics))
    
    
def run_tf(config, train_df, valid_df, test_df):
    # getting setups
    mname = config["meta"]["model_name"]
    features = config["data_columns"]["gen3_features"]
    features = [f+"_xf" for f in features]
    params = config["model_params"]
    target_col = config["data_columns"]["target_cols"][0]

    # setup model
    from src.architectures import get_mlp_clf, get_initial_bias
    import tensorflow as tf
    from tensorflow import keras
    
    n_features = params["n_features"]
    n_classes = params["n_classes"]
    nhids = params["nhids"]    
    if params["init_bias_by_target"]:
        output_bias = get_initial_bias(train_df[target_col])
    else:
        output_bias = None
    log_er.info(f"initial bias : {output_bias}")
    model = get_mlp_clf(n_features, n_classes, nhids, output_bias=output_bias)
    
    if params["optim"] == "adam":
        if params["lr_decay_steps"] is not None and params["lr_decay_rate"] is not None:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=params["lr"],
                decay_steps=params["lr_decay_steps"],
                decay_rate=params["lr_decay_rate"],
            )
        else: lr_schedule=params["lr"]
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        opt = params["optim"]
    model.compile(optimizer=opt,
                  loss=params["loss"],
                  metrics=params["metrics"]) 
    
    # get data ready
    trainer = TFTrainer()
    
    y_train = tf.keras.utils.to_categorical(train_df[target_col], n_classes)
    y_valid = tf.keras.utils.to_categorical(valid_df[target_col], n_classes)
    y_test = tf.keras.utils.to_categorical(test_df[target_col], n_classes)

    train_data = trainer.df_to_tensor(train_df[features], y_train, batch_size=params["batch_size"])
    valid_data = trainer.df_to_tensor(valid_df[features], y_valid, batch_size=params["batch_size"], shuffle=False)
    test_data = trainer.df_to_tensor(test_df[features], batch_size=params["batch_size"], shuffle=False)
    
    # let tensorboard take care of logginer
        
    # setup trainer
    es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                       patience=params["patience"],
                                       restore_best_weights=True)
    class_weight = None
    if "class_weight" in params and params["class_weight"] is not None:
        class_weight = dict((int(k), v) for k,v in params["class_weight"].items())   
        log_er.info(str(class_weight))
        
    callbacks = [es]
    history = trainer.train(model, 
                  train_data,
                  epochs=params["epochs"],
                  validation_data=valid_data,
                  callbacks=callbacks,
                  class_weight=class_weight,
                 )

    log_er.info("model fitted")

    save_path = f"./models/{mname}/model"
    trainer.save_model(save_path)
    log_er.info("model saved at: " + save_path)

    pred_test = trainer.predict(test_data)[:,1]
    test_df["pred"] = pred_test

    metrics = get_pred_reports(test_df, target_col, ['pred'])
    metrics.to_csv(f"./models/{mname}/metrics.csv")
    log_er.info(str(metrics))
    
    with open(f"./models/{mname}/config.json", "w") as f:
        json.dump(params, f, indent=4)
    pd.DataFrame(history.history).to_csv(f"./models/{mname}/history.csv")
    
    log_er.info(model.summary())
    log_er.info("metrics: " + str(metrics))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Commandline interface for the Pipeline module"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.json",
        help="path to config.json file",
        dest="config",
    )
    parser.add_argument(
        "-d",
        "--debug",
        default=None,
        help="Debug mode, run with small data set with provided size",
        dest="debug",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.debug)
