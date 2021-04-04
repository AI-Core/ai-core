import torch
import requests
import os
import json
from PIL import Image, UnidentifiedImageError

def download_file(src_url, local_destination):
    response = requests.get(src_url)
    with open(local_destination, 'wb+') as f:
        f.write(response.content)

def create_id_from_url(url):
    id = url.split('/')[-1] # get filename
    id = id.strip('\n') # strip trailing newlines
    id = id.split('?')[0] # remove trailing query string params
    id = id.split('&')[0] # remove trailing query string params
    if len(id.split('.')) == 1: # if no file extension in image id
        id += '.jpg' # guess and add .jpg
    return id

img_dir = 'images'
if not os.path.exists(img_dir):
    os.mkdir(img_dir)

shoe_files = os.listdir('shoes_urls')[1:]
for file in shoe_files:
    print(file)
    with open(f'shoes_urls/{file}') as f:
        shoes = f.readlines()
    # print(shoes)
    for url in shoes:
        print(url)

        brand_dir = f'{img_dir}/{file.split(".")[0]}'
        if not os.path.exists(brand_dir):
            os.mkdir(brand_dir)

        id = create_id_from_url(url) # TODO replace with uuid

        if 'https://pumaimages.azureedge.net' in url:
            print('skipping shit website')
            continue

        if '.tif' in id:
            print('skipping .tif file')
            continue

        print(id)
        local_dest = f'{brand_dir}/{id}'
        if os.path.exists(local_dest):
            print('image already exists')
            continue
        try:
            download_file(url, local_dest)
        except requests.exceptions.ConnectionError:
            print('failed to get url')
            continue
        try:
            Image.open(local_dest).close() # open and close to check it works
            print('image opened')
        except UnidentifiedImageError:
            print('img wasnt opened')
            os.remove(local_dest)
            continue
        print()
        # sdfvsd
scds


class Shoes():
    def __init__(self, root_dir='images', download=False, transform=None):
        if download:
            r = requests.get('s3://') # download zip file
            # unzip into root_dir
        
        self.img_fps = (f'{root_dir}/{brand_dir}/{filename}' for brand_dir in os.listdir(root_dir) for filename in os.listdir(f'{root_dir}/{brand_dir}')) # generate image filepaths

    def __len__(self):
        return len(self.img_fps)

    def __getitem__(self, idx):
        fp = self.img_fps[idx]
        img = Image.open(fp)
        if self.transform:
            img = self.transform(img)
        return img

