#!/bin/bash

# setup virtual environment
ENVIRONMENT=rdsenv
conda create -y -n ENVIRONMENT python=3.7
conda install nb_conda_kernels ipykernel --name ENVIRONMENT --yes

# # install packages
# source activate "$ENVIRONMENT"
# pip install -i https://repository.sofi.com/artifactory/api/pypi/pypi/simple -r requirements.txt --use-deprecated=legacy-resolver
# conda deactivate

# # install kernel
# source activate JupyterSystemEnv
# python -m ipykernel install --user --name $ENVIRONMENT --display-name "$ENVIRONMENT"
