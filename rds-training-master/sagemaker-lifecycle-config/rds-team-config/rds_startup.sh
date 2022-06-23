#!/bin/bash

# Install SDM
cd /tmp
rm -rf sdm*
curl -s -J -O -L https://app.strongdm.com/releases/cli/linux
unzip sdmcli*.zip
chmod +x sdm
sudo mv sdm /usr/local/bin

# create sdm-init.sh script
SDM_SCRIPT=/home/ec2-user/sdm-init.sh
cat >${SDM_SCRIPT} <<EOL
#!/bin/bash
echo "Init SDM Connections ..."
nohup sdm listen -d &>/home/ec2-user/sdm_listen.log &
sleep 5
sdm login
sdm connect -a
sdm status
EOL
chmod a+x ${SDM_SCRIPT}

# execute user configured scripts
START_INITD_DIR=/home/ec2-user/SageMaker/init.d
mkdir -p ${START_INITD_DIR}
for f in ${START_INITD_DIR}/*; do
	case "$f" in
		*.sh)
			if [ -x "$f" ]; then
				echo "$0: running $f"
				"$f"
			else
				echo "$0: sourcing $f"
				. "$f"
			fi
			;;
		*)  
		    echo "$0: ignoring $f" ;;
	esac
	echo
done


# Install all the DB packages
sudo yum -y install --skip-broken python36u-devel mysql-devel postgres-devel mysql57-libs-5.7.24-1.10.amzn1.x86_64  mysql57.x86_64 postgresql96.x86_64


# Extra config.

set -e

sudo -u ec2-user -i <<'EOF'

# set up conda
conda update -y -n root conda
conda update -y --all


# create environments 
conda create -y -n ml_basic_py37 python=3.7
# eventually - pytorch env, deployment env, etc.

#pip install packagess
ENVIRONMENT=ml_basic_py37
PACKAGES="pandas scikit-learn numpy lightgbm mdsutils tqdm matplotlib seaborn s3fs pyarrow"

#make things work with conda
conda install nb_conda_kernels ipykernel --name $ENVIRONMENT --yes

source activate "$ENVIRONMENT"
pip install --index-url https://build.sofi.com/artifactory/api/pypi/pypi/simple --upgrade $PACKAGES --use-deprecated=legacy-resolver
#hack for pyarrow

conda deactivate

# JupyerLab Extensions, install kernels
source activate JupyterSystemEnv

# install kernel
python -m ipykernel install --user --name $ENVIRONMENT --display-name "$ENVIRONMENT"

# yarn fix
/home/ec2-user/anaconda3/envs/JupyterSystemEnv/bin/npm install -g yarn
cp /home/ec2-user/anaconda3/envs/JupyterSystemEnv/lib/node_modules/yarn/bin/yarn.js /home/ec2-user/anaconda3/lib/python3.6/site-packages/jupyterlab/staging/yarn.js

# JupyterLab Themes

jupyter labextension install @oriolmirosa/jupyterlab_materialdarker
jupyter labextension install jupyterlab_filetree
jupyter labextension install jupyterlab-python-file
jupyter labextension install jupyterlab-recents
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install @jupyterlab/toc
jupyter labextension install @karosc/jupyterlab_dracula

conda deactivate

# Git Please change this part

cp ~/SageMaker/keys/* ~/.ssh/
git config --global user.name jxu-sofi
git config --global user.email jxu@sofi.org

EOF