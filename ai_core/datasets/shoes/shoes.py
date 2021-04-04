import torch
import requests
import os
import json
from PIL import Image

def download_file(src_url, local_destination):
    response = requests.get(src_url)
    with open(local_destination, 'wb+') as f:
        f.write(response.content)

img_dir = 'shoes_imgs'
if not os.path.exists(img_dir):
    os.mkdir(img_dir)

shoe_files = os.listdir('shoes_urls')
for file in shoe_files:
    print(file)
    with open(f'shoes_urls/{file}') as f:
        shoes = f.readlines()
    print(shoes)
    for url in shoes:
        print(url)
        id = url.split('/')[-1].strip('\n')
        print(id)
        local_dest = f'{img_dir}/{id}'
        download_file(url, local_dest)
        try:
            Image.open(local_dest).close() # open and close to check it works
            print('image opened')

        except:
            print('img wasnt opened')
            os.remove(local_dest)
        print()
        # sdfvsd
scds


class Shoes():
    def __init__(self, root_dir='.', download=False):
        if download:
            requests.get('s3://')
        self.images = imgs

