# zip up images

# upload to s3
# last img = https://resources.mandmdirect.com/Images/_default/l/c/1/lc1392_1_cloudzoom.jpg

import shutil
import boto3

def zip_up(zip_name, dir_name):
    shutil.make_archive(zip_name, 'zip', dir_name)

def s3_upload(bucket_name, file_name):
    s3 = boto3.client('s3')
    object_name = 'data.zip'
    response = s3.upload_file(file_name, bucket_name, object_name)
