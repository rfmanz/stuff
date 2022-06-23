#! /bin/bash
 

# put your folder path here
cur_path=/home/ec2-user/SageMaker/ds_train/logging

# create a folder
mkdir -p ${cur_path}/logs
logfile=${cur_path}/logs/p000.log

# location for your sql file 
sql_path=${cur_path}/sql_file



#mkdir -p ${model_path}/logs

if test -f "$logfile"; then
    rm $logfile
fi


cat << EOF >> $logfile
	Pulling data for target profiling 
EOF


python p000_logging.py \
--SQL_PATH "${sql_path}" \
1>> $logfile 2>> $logfile


RC=$?

if test $RC -ne 0 ; then
	echo "Data pull failed" 
	exit 106
fi
