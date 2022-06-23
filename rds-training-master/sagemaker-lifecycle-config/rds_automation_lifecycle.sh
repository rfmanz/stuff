PYTHON_HOME=/home/ec2-user/anaconda3/envs/python3/bin
echo "export AIRFLOW_HOME=/home/ec2-user/SageMaker/airflow" >> /home/ec2-user/.bashrc
set -e
AIRFLOW_HOME=/home/ec2-user/SageMaker/airflow
docker run --name airflow_postgres -p 5432:5432 -e POSTGRES_HOST_AUTH_METHOD=trust -d -v /home/ec2-user/SageMaker/airflow_backend/:/var/lib/postgresql/data postgres:11
$PYTHON_HOME/pip uninstall -y enum34
$PYTHON_HOME/pip install apache-airflow==1.10.5
$PYTHON_HOME/pip install flask==1.1.0 werkzeug==0.16.1
$PYTHON_HOME/airflow initdb
sudo yum -y install --skip-broken python36u-devel mysql-devel postgres-devel mysql57-libs-5.7.24-1.10.amzn1.x86_64  mysql57.x86_64 postgresql96.x86_64
$PYTHON_HOME/pip install paramiko psycopg2-binary mysqlclient==1.3.12 sshtunnel pysftp
$PYTHON_HOME/pip install fastparquet lightgbm pdpbox SQLAlchemy==1.3.15
$PYTHON_HOME/pip install -U scikit-learn
$PYTHON_HOME/pip install smart_open column
$PYTHON_HOME/pip install --upgrade oauth2client
$PYTHON_HOME/pip install -U pyarrow
$PYTHON_HOME/pip install --index-url https://repository.sofi.com/artifactory/api/pypi/pypi/simple mdsutils
$PYTHON_HOME/pip install --index-url https://repository.sofi.com/artifactory/api/pypi/pypi/simple sofi_dsutils
$PYTHON_HOME/pip install --upgrade snowflake-connector-python
$PYTHON_HOME/pip install snowflake-sqlalchemy