#!/bin/bash

# copy the startup.sh script from s3
aws s3 cp s3://sofi-data-science/jxu/startup.sh /home/ec2-user/SageMaker/

# change the access permission of the copied script
chmod +x /home/ec2-user/SageMaker/startup.sh

# run the script in the background
nohup /home/ec2-user/SageMaker/startup.sh &