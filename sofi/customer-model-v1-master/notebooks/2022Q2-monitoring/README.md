### Monitoring
---

Manual: please go through the following steps to conduct model monitoring.

1. duplicate this directory, named as `YYYYQn-monitoring`, in the `notebooks` directory
    * need to make sure `config.json` and `artifacts` are in the parent directory of `notebooks`
2. update information and generate a new `config.json` by following the steps in `0-setup.ipynb`
3. after verifying the information is correct, either
    * iterate through the notebooks in the provided order, or
    * run the `run_monitoring.sh`
This should generate everything needed to the `artifact` folder. You can find the relative path to the artifact folder by looking at searching for it in the config file. `artifact_path`