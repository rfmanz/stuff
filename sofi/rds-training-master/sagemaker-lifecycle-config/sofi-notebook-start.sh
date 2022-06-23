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
#sudo pip install --upgrade pip
#sudo pip3 install psycopg2-binary mysqlclient


