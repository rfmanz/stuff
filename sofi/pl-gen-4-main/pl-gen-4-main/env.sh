#!/bin/bash

# install packages in tensorflow2_p36
# source activate "amazonei_tensorflow2_p36"
# pip install -i https://repository.sofi.com/artifactory/api/pypi/pypi/simple -r requirements.txt
# conda deactivate

# setup virtual environment
ENVIRONMENT=pl_gen4
conda create -y -n $ENVIRONMENT python=3.7
conda install nb_conda_kernels ipykernel --name $ENVIRONMENT --yes

# install packages
source activate "$ENVIRONMENT"
pip install --upgrade pip
pip install -i https://repository.sofi.com/artifactory/api/pypi/pypi/simple -r requirements.txt
conda deactivate

# install kernel
source activate JupyterSystemEnv
python -m ipykernel install --user --name $ENVIRONMENT --display-name "$ENVIRONMENT"

jupyter labextension install jupyterlab-plotly

# install autogluon
# source activate "mxnet_p36"
# pip install --upgrade pip "mxnet<2.0.0"
# pip install bokeh==2.0.1
# pip install autogluon
# pip install -i https://repository.sofi.com/artifactory/api/pypi/pypi/simple -r requirements.txt
# conda deactivate