import json, os, sys
import pandas as pd
import numpy as np
import pickle as pkl

from src.dataloader import Dataloader
import src.train as train
import src.governance as gvrn
import src.preprocessing.customer_utils as cu
import src.preprocessing.utils as pu


def generate_new_project(project_name):
    """ Generate a new project in the transactions_dataloader directory """
    import os, shutil

    path = f"../projects_dataloader/{project_name}"

    # validation for name duplication
    exists = os.path.isdir(path)
    if exists:
        print("project already exist")

    # generate from current directory
    dirname, filename = os.path.split(os.path.abspath(__file__))

    # move src
    shutil.copytree(os.path.join(dirname, "src"), os.path.join(path, "src"))

    # move files in parent dir over
    files = [".gitignore", "debug_ids.csv", "env.sh", "requirements.txt", "main.py"]
    for file in files:
        shutil.copy2(file, os.path.join(path, file))
    # copy empty config.json over
    shutil.copy2("config_example.json", os.path.join(path, "config.json"))

    # replace example with the name - do not need yet
    # fin = open("{}/__init__.py".format(path), "rt")
    # data = fin.read()
    # data = data.replace('ExampleStrategy', name)
    # fin.close()
    # fin = open("{}/__init__.py".format(path), "wt")
    # fin.write(data)
    # fin.close()

    print(f"Project created at: {path}")


def train_main(config_path):
    
    with open(config_path,"r") as f:
        config = json.load(f)

    SEED = config["model_params"]["seed"]
    TARGET_COL = "target"
    INDETERMINATE_COL = config["indeterminate_col"]

    # load data
    print("-------------- loading data --------------")
    modeling_df, valid_dfs, test_dfs = train.prep_data(config, TARGET_COL)

    # set context
    print("-------------- setting context --------------")
    base_dir = config["base_dir"]
    artifact_path = config["artifact_path"]
    govn_path = os.path.join(base_dir, artifact_path, "governance") 

    context = {"modeling_df": modeling_df,
               "valid_dfs": valid_dfs,
               "test_dfs": test_dfs,
               "config": config,
               "target_col": "target",
               "date_col": "transaction_datetime",
               "seed": config["model_params"]["seed"],
               "indeterminate_col": config["indeterminate_col"],
               "model_name": config["model_name"],
               "govn_path": govn_path,
               "pred_cols": []}
  
    # setting up preprocess fns
    print("-------------- baselines --------------")
    context["baseline_models"] = train.get_baseline_models(config)
    context["baseline_models"]["deposit_v1"]["preprocess"] = None
    context["baseline_models"]["customer"]["preprocess"] = cu.preprocess
    context["baseline_models"]["customer_refit_2021Q1"]["preprocess"] = cu.preprocess
    context["baseline_models"]["deposit_v2_ach_dev_final"]["preprocess"] = pu.preprocess

    # baselines
    result = train.get_baseline_preds(context["baseline_models"],
                                                          modeling_df,
                                                          valid_dfs,
                                                          test_dfs)
    modeling_df, valid_dfs, test_dfs, new_cols = result
    context["pred_cols"].extend(new_cols)

    # train
    print("-------------- train - valid --------------")
    config, context = train.train_model(config, context, pu.preprocess)
    config, context = train.validate_model(config, context, pu.preprocess)

    # get predictions
    clf = context["model_object"]
    result = train.get_model_preds(clf, modeling_df, 
                                   valid_dfs, test_dfs, pu.preprocess)
    modeling_df, valid_dfs, test_dfs, new_cols = result
    context["pred_cols"].extend(new_cols)
    
    # get governance
    print("-------------- governance --------------")
    gvrn.save_governance_data(config, context, pu.preprocess)
    
    # save config to folder
    with open(os.path.join(base_dir, artifact_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    
def get_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Command line interface to ACH deposit model v2."
    )
    parser.add_argument("-n", "--new_project", type=str, default=None, help="Name of new project")
    parser.add_argument(
        "-r",
        "--raw",
        action="store_true",
        default=False,
        help="Pull raw data files.",
        dest="gr",
    )
    parser.add_argument(
        "-p",
        "--processed",
        action="store_true",
        default=False,
        help="Build processed data file.",
        dest="gp",
    )
    parser.add_argument(
        "-j",
        "--join",
        action="store_true",
        default=False,
        help="Join processed data.",
        dest="gj",
    )
    parser.add_argument(
        "-f",
        "--features",
        action="store_true",
        default=False,
        help="Build features from processed+joined data. Pass path to processed data file.",
        dest="gf",
    )
    parser.add_argument(
        "-l",
        "--labels",
        action="store_true",
        default=False,
        help="Add labels to processed/featurized dataframe.",
        dest="gl",
    )
    parser.add_argument(
        "-pp",
        "--postprocess",
        action="store_true",
        default=False,
        help="Postprocess, after labeling and before model training.",
        dest="gpp"
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="""Enter debugging mode by only process on the provided ids. 
                       Note all data still need to be loaded""",
        dest="debug",
    )
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        default=False,
        help="train model",
        dest="gt",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.json",
        help="path to config.json",
        dest="gc",
    )
    parser.add_argument(
        "-cl",
        "--clean",
        default=None,
        help='Please choose between "remove_non_current" and "remove_current"',
        dest="gclean",
    )
    
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    
    args = get_args()
    
    if args.new_project:
        generate_new_project(args.new_project)

    dl = Dataloader(debug=args.debug) 
    
    if args.gr:
        # get raw data
        dl.query(prefix="raw", debug=args.debug)
        
    if args.gp:
        # get processed data
        dl.process()
        
    if args.gj:
        # get join
        dl.join()
        
    if args.gf:
        # get features
        dl.features()

    if args.gl:
        # get labels
        dl.labels()
        
    if args.gpp:
        # get post processing
        raise NotImplemented
        
    if args.gt:
        # train
        train_main(args.gc)
        
    if args.gclean:
        dl.clean(args.gclean)
        
        

