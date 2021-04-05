import torch
import requests
import os
from PIL import Image
import shutil

class Shoes():
    def __init__(self, root_dir='images', download=False, transform=None):
        self.transform = transform
        if download:
            # TODO dont download if it already exists
            zip_dir = f'{root_dir}.zip'
            response = requests.get('https://ai-core-shoes-dataset.s3.eu-west-2.amazonaws.com/data.zip') # download zip file
            with open(zip_dir, 'wb') as f:
                f.write(response.content) # write data to zip file
            shutil.unpack_archive(zip_dir, root_dir) # unzip into root_dir
            os.remove(zip_dir)# remove zip file
        try:
            self.img_fps = [f'{root_dir}/{brand_dir}/{filename}' for brand_dir in os.listdir(root_dir) for filename in os.listdir(f'{root_dir}/{brand_dir}')] # generate image filepaths
        except FileNotFoundError:
            raise FileNotFoundError('You need to download the dataset by using download=True')

    def __len__(self):
        return len(self.img_fps)

    def __getitem__(self, idx):
        fp = self.img_fps[idx]
        img = Image.open(fp)
        if self.transform:
            img = self.transform(img)
        return img

if __name__ == '__main__':
    import random
    shoes = Shoes(download=False)
    print(len(shoes))
    # img = shoes[random.randint(0, len(shoes))]
    # img.show()