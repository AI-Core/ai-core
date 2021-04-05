import torch
import requests
import os
import json
from PIL import Image, UnidentifiedImageError
import sys
sys.path.append('../utils')
import create

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

def get_data(idx):
    img_dir = 'images'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    shoe_files = os.listdir('shoes_urls')[idx]
    shoe_files = [shoe_files]
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
            except requests.exceptions.TooManyRedirects:
                print('too many redirects')
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

if __name__ == '__main__':
    # get_data()

    create.zip_up('/Users/ice/shoes', 'images')
    create.s3_upload('ai-core-shoes-dataset', '/Users/ice/shoes.zip')
    

