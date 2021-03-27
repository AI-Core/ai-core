from PIL import Image
from PIL import ImageFile
import torch
import torchvision.transforms as transforms
import glob
import os
import random
import json
import requests
from pathlib import Path
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def tmp_func(x):
    return x.repeat(3, 1, 1)

class Furniture(torch.utils.data.Dataset):
    '''
    The ImageDataset object inherits its methods from the
    torch.utils.data.Dataset module.
    It loads all images from an image folder, creating labels
    for each image according to the subfolder containing the image
    and transform the images so they can be used in a model
    Parameters
    ----------
    root_dir: str
        The directory with subfolders containing the images
    transform: torchvision.transforms 
        The transformation or list of transformations to be 
        done to the image. If no transform is passed,
        the class will do a generic transformation to
        resize, convert it to a tensor, and normalize the numbers
    
    Attributes
    ----------
    files: list
        List with the directory of all images
    labels: set
        Contains the label of each sample
    dict_encoder: dict
        Dictionary to translate the label to a 
        numeric value
    '''
    def __init__(self, root_dir, transform=True, download=True):
        
        self.root_dir = root_dir
        if download:
            self.download(self.root_dir)
        else:
            if not os.path.exists(root_dir):
                raise RuntimeError('Dataset not found.' +
                                   'You can use download=True to download it')

        self.files = glob.glob(os.path.join(f'{self.root_dir}', '*', '*', '*' + '.*'))
        self.labels = list(set([fp.split(os.sep)[1] for fp in self.files]))
        self.num_classes = len(self.labels)
        self.dict_encoder = {y: x for (x, y) in enumerate(self.labels)}
        self.transform = transform
        if transform is True:
            self.transform = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) # GANHACK #1
            ])
            
            self.transform_Gray = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Lambda(tmp_func),
            transforms.Normalize((0.5,), (0.5,))
            ])

    def __getitem__(self, index):

        img_name = self.files[index]
        label = img_name.split(os.sep)[1]
        label = self.dict_encoder[label]
        label = torch.as_tensor(label)
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.files)
    
    def download(self, root):

        with open('./furniture_data.json') as f:
            data = json.load(f) # read data containing image paths

        paths = ('/'.join(path.split('/')[1:]) for category in data.values() for item in category.values() for path in item['images']) # generate paths

        for path in tqdm(paths):
            # OS FRIENDLY WAYS TO GET THE IMG PATH AND DIR
            fp = os.path.join(self.root_dir, *os.path.split(path))
            if os.path.exists(fp):
                continue
            dir = os.path.join(*os.path.split(fp)[:1])
            Path(dir).mkdir(parents=True, exist_ok=True) # create dir if doesnt exist
            response = requests.get(f'https://ikea-dataset.s3.us-east-2.amazonaws.com/data/{path}')
            with open(fp, 'wb') as f:
                f.write(response.content)

def split_train_test(dataset, train_percentage):
    train_split = int(len(dataset) * train_percentage)
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_split, len(dataset) - train_split]
    )
    return train_dataset, validation_dataset

if __name__ == '__main__':
    dataset = Furniture(root_dir='images', download=True, transform=False)

    # GIVE RANDOM EXAMPLE
    while True:
        idx = random.randint(0, len(dataset))
        img, label = dataset[idx]
        print(label)
        img.show()
        break # your computer will die if you remove this