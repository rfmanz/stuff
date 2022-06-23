#! /bin/bash
 

data_path=/home/ec2-user/SageMaker/template/Data
cur_path=/home/ec2-user/SageMaker/ds_train/logging
logfile=${cur_path}/logs/p003_save_to_s3.log


mkdir -p ${cur_path}/logs

if test -f "$logfile"; then
    rm $logfile
fi


cat << EOF >> $logfile
	Save the files to s3://sofi-data-science/gen2_collection_model/
EOF


python p003_save_to_s3.py \
--DATA_PATH "${data_path}/f000_pull_data/" \
1>> $logfile 2>> $logfile


RC=$?

if test $RC -ne 0 ; then
	echo "Save to s3 failed" 
	exit 106
fi
