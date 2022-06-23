aws s3 cp s3://sofi-data-science/Risk_DS/scripts/rds_startup.sh /home/ec2-user/SageMaker/
chmod +x /home/ec2-user/SageMaker/rds_startup.sh
nohup /home/ec2-user/SageMaker/rds_startup.sh &