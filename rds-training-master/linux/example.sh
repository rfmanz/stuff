#! /bin/bash

logfile=${PWD}

cat << EOF >> $logfile
	example for running py.file in script
EOF

python -u $filename.py


RC=$?


if test $RC -ne 0 ; then
	echo "training model failed" 
	exit 106
fi




