import sys
sys.path.append('.')
# if __name__ == '__main__':

from ai_core.datasets import Furniture
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from ai_core.models.gans import CWGAN
dataset = Furniture('./images', download=False)
print(len(dataset))
# Number of workers for dataloader
workers = 2
batch_size = 64
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

cwgan = CWGAN()
cwgan.fit()