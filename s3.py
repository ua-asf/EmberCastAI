import boto3
import numpy
import io
import os
import logging

log = logging.getLogger(__name__)

def download_file ( s3_bucket, s3_key ):

    file_name = s3_key.split("/")[-1]
    local_file_name = f'/tmp/{file_name}'

    if not os.path.exists(local_file_name):
        log.info(f"Downloading {local_file_name} from s3://{s3_bucket}/{s3_key}")
        with open(local_file_name, 'wb') as f:
            boto3.client("s3").download_fileobj(s3_bucket, s3_key, f)
    else:
        log.info(f"Reusing local {local_file_name}")

    return local_file_name

def get_np_file ( s3_bucket, s3_key, stream=False):
    if stream:
        return get_np_file_stream( s3_bucket, s3_key )

    local_file_name = download_file ( s3_bucket, s3_key )

    return numpy.load(local_file_name)

def get_np_file_stream( s3_bucket, s3_key ):
    obj = boto3.resource("s3").Object(s3_bucket, s3_key)
    log.info(f"Streaming s3://{s3_bucket}/{s3_key}")
    with io.BytesIO(obj.get()["Body"].read()) as f:
        f.seek(0)
        return numpy.load(f)


