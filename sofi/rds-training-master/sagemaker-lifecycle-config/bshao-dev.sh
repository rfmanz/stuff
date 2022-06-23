#!/bin/bash

set -e

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


#Change time to PST
echo "ZONE="America/Los_Angeles" UTC=true" | sudo tee /etc/sysconfig/cloc
sudo ln -sf /usr/share/zoneinfo/America/Los_Angeles /etc/localtime

# Install all the DB packages
sudo yum -y install --skip-broken python36u-devel mysql-devel postgres-devel mysql57-libs-5.7.24-1.10.amzn1.x86_64  mysql57.x86_64 postgresql96.x86_64
sudo pip install --upgrade pip

# Configure Python3 env
PYTHON_HOME=/home/ec2-user/anaconda3/envs/python3/bin
$PYTHON_HOME/pip install --upgrade pip
$PYTHON_HOME/pip install --upgrade scikit-learn
$PYTHON_HOME/pip install feather-format
$PYTHON_HOME/pip install tqdm
$PYTHON_HOME/pip install dill
$PYTHON_HOME/pip install pandas
$PYTHON_HOME/pip install lightgbm==2.3.1
$PYTHON_HOME/pip install pandasql
$PYTHON_HOME/pip install lifelines
$PYTHON_HOME/pip install xgboost
$PYTHON_HOME/pip install verify
$PYTHON_HOME/pip install mysql-connector-python
$PYTHON_HOME/pip install xmltodict
$PYTHON_HOME/pip install eli5
$PYTHON_HOME/pip install hjson
$PYTHON_HOME/pip install qgrid
$PYTHON_HOME/pip install varclushi
$PYTHON_HOME/pip install psycopg2-binary
$PYTHON_HOME/pip install --upgrade snowflake-connector-python
#$PYTHON_HOME/pip install mysqlclient