import click, mlflow
import sys, os, json, argparse
import numpy as np
import pandas as pd

@click.command()
@click.option("--config", default="./config.json", type=str)
@click.option("--debug", default=False, type=bool)

def main(config, debug):
    # get dataloader
    
    # get Trainer

    # if validate - produce valid result

    # if test - produce test result

    # auxiliary
    return



if __name__ == "__main__":
    main()
    
    
