"""
aws.py

AWS helper functions.
"""

import boto3
import botocore
import s3fs


def download_s3(bucket_name, path_s3, path_local, aws_access_key_id=None, 
                aws_secret_access_key=None, aws_session_token=None, region_name=None):
    """Download file from S3.
    
    Parameters
    ----------
    bucket_name : string
        Bucket from which to download the file.
    path_s3 : 
        Path to file in S3 bucket.
    path_local:
        Local path where the file will be downloaded.
        
    Returns
    -------
    """
    boto3.setup_default_session(aws_access_key_id=aws_access_key_id,
                                aws_secret_access_key=aws_secret_access_key,
                                aws_session_token=None, region_name=None)
    
    bucket = boto3.resource("s3").Bucket(bucket_name)
    
    try:
        bucket.download_file(path_s3, path_local)
    except botocore.exceptions.ClientError as e:
        raise


def upload_s3(bucket_name, path_local, path_s3, aws_access_key_id=None, 
              aws_secret_access_key=None, aws_session_token=None, region_name=None):
    """Upload local file to S3.
    
    Parameters
    ----------
    bucket_name : string
        Bucket from which to download the file.
    path_local:
        Local path where to the file that will be uploaded.
    path_s3 : 
        Path where the file will be uploaded in S3 bucket.
        
    Returns
    -------
    """
    boto3.setup_default_session(aws_access_key_id=aws_access_key_id,
                                aws_secret_access_key=aws_secret_access_key,
                                aws_session_token=None, region_name=None)
    
    with open(path_local, 'rb') as file:
        bucket = boto3.resource("s3").Bucket(bucket_name)
        bucket.put_object(Key=path_s3, Body=file)

        
def python_object_to_s3(obj, bucket_name, path_s3, is_bytes=False, **kwargs):
    """Upload python object to S3.
    
    This function is a WIP.
    
    Parameters
    ----------
    obj:
        This is either an object formatted as bytes, or a function that
        takes an output files as an argument.
    bucket_name : string
        Bucket from which to download the file.
    path_s3 : 
        Path to file in S3 bucket.
        
    Returns
    -------
    """
    s3 = s3fs.S3FileSystem(anon=False)

    if is_bytes:
        with s3.open('{}/{}'.format(bucket_name, path_s3), 'w') as f:
            f.write(obj)
    else:
        with s3.open('{}/{}'.format(bucket_name, path_s3), 'w') as f:
            obj(f, **kwargs)

            
def pandas_df_to_s3(df, bucket, path_s3, file_format='parquet', **kwargs):
    """
    Upload Pandas DataFrame to S3
    

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to upload
    bucket: string
        Bucket from which to upload the file: e.g. sofi-data-science
    path_s3: string
        path_s3 to file in s3 bucket
    file_format: string
        Format to store the df in, currently supports feather, parquet, and csv.

    Returns
    -------
    """
    import s3fs, boto3
    boto3.setup_default_session()
    s3 = s3fs.S3FileSystem(anon=False)
    
    if file_format in ['feather', 'parquet']:
        with s3.open(f'{bucket}/{path_s3}', 'wb') as f:
            getattr(df, f'to_{file_format}')(f, **kwargs)
    elif file_format in ['csv']:
        with s3.open(f'{bucket}/{path_s3}', 'w') as f:
            getattr(df, f'to_{file_format}')(f, **kwargs)
    else:
        raise NotImplemented
        