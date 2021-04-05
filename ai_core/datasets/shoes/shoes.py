import torch
import requests
import os
from PIL import Image

class Shoes():
    def __init__(self, root_dir='images', download=False, transform=None):
        self.transform = transform
        if download:
            r = requests.get('s3://') # download zip file
            # unzip into root_dir
        
        self.img_fps = [f'{root_dir}/{brand_dir}/{filename}' for brand_dir in os.listdir(root_dir) for filename in os.listdir(f'{root_dir}/{brand_dir}')] # generate image filepaths

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
    shoes = Shoes()
    print(len(shoes))
    img = shoes[random.randint(0, len(shoes))]
    img.show()