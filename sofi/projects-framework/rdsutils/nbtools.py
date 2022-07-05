# Author: Marketing Data Science
# Vinayak Raja

import boto3
from time import sleep
from airflow.contrib.hooks.ssh_hook import SSHHook
from airflow.contrib.operators.ssh_operator import SSHOperator
from botocore.exceptions import ClientError

class NBManager():
    def __init__(self, nbname):
        self.nbname=nbname
        self.sm = boto3.client('sagemaker')
        self.ec2 = boto3.client('ec2')
        self.status = self.wait_notebook()
        if self.status=='InService':
            self.ip = self._fetch_nb_ip()
        
    def wait_notebook(self, target=None):
        wait_flag = False
        while True:
            resp = self.sm.describe_notebook_instance(NotebookInstanceName=self.nbname)
            status = resp['NotebookInstanceStatus']
            if status in ['Starting', 'Stopping', 'Pending']:
                if not wait_flag:
                    print("Waiting for notebook to settle...")
                    wait_flag=True
                sleep(15)
            elif target and target!=status:
                raise ClientError(error_response={'Error':{"Code":'Notebook in unrecoverable state: {}'.format(status=status)}},operation_name="wait_notebook")
            else: 
                return status
    
    def start_notebook(self):
        if self.status=='Stopped':
            print("Starting {nbname}".format(nbname=self.nbname))
            self.sm.start_notebook_instance(NotebookInstanceName=self.nbname)
            self.status = self.wait_notebook(target='InService')
        elif self.status=='InService':
            print("Notebook is in service")
        else: 
            raise ClientError(error_response={'Error':{"Code":'Notebook in unrecoverable state: {status}'.format(status=self.status)}},operation_name="start_notebook")
        self.ip = self._fetch_nb_ip()
    
    def stop_notebook(self):
        print("Stopping notebook {nbname}".format(nbname=self.nbname))
        if self.status=='InService':
            self.sm.stop_notebook_instance(NotebookInstanceName=self.nbname)
            self.status = self.wait_notebook(target='Stopped')
        elif self.status=='Stopped':
            print("Notebook is stopped")
        else: 
            raise ClientError(error_response={'Error':{"Code":'Notebook in unrecoverable state: {status}'.format(status=self.status)}},operation_name="stop_notebook")
    
    def _fetch_nb_ip(self):
        eni = self.sm.describe_notebook_instance(NotebookInstanceName=self.nbname)['NetworkInterfaceId']
        for interface in self.ec2.describe_network_interfaces(NetworkInterfaceIds=(eni,))['NetworkInterfaces']:
            ip = interface['PrivateIpAddress']
            if '10.220' in ip:
                return ip
    
    def send_ssh(self, command):
        if self.ip is not None:
            ssh = SSHHook(remote_host=self.ip, username='ec2-user', key_file="/home/ec2-user/.ssh/id_rsa")
            ssh.allow_host_key_change=True
            ssh = SSHOperator(task_id="test",ssh_hook=ssh, command=command)
            ssh.execute(context=None)
