import sagemaker
import os
import boto3
from botocore.exceptions import ClientError
import botocore

def upload_to_s3(bucket, local_path, s3_path):
    session = sagemaker.Session()
    
    filepath = session.upload_data(bucket = bucket, 
                                   path = local_path, 
                                   key_prefix = s3_path)
    return filepath


def object_exists(bucket, key):
    s3 = boto3.resource('s3')

    try:
        s3.Object(bucket, key).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise str(e)
    else:
        return True
    
def download_s3_file(bucket_name, s3_key, local_filename):
    """
    Downloads an s3 file to the tmp directory
    """

    s3 = boto3.client("s3")

    try:
        with open(local_filename, "wb") as f:
            s3.download_fileobj(bucket_name, s3_key, f)
    
    except ClientError as e:
        print(e)
        return None
    
    
def unzip(path):
    import zipfile
    
    unziped_path = path.replace('.zip','')
    with zipfile.ZipFile(path,"r") as zip_ref:
        zip_ref.extractall(unziped_path)
        
    os.remove(path)
    return unziped_path
    