import torch
import sys
sys.path.append('.')
import utils
from CNN import CNN 
import numpy as np
import torch.nn.functional as F
from ray import tune
import os
from train import train
from torch.utils.data import DataLoader
from NN import NN
from TransposeCNN import TransposeCNN

batch_size = 16

train_data, val_data, test_data = utils.get_splits()

class ConvAutoencoder(torch.nn.Module):
    def __init__(
            self, 
            encoder_channels,
            encoder_linear_layers,
            decoder_channels,
            decoder_linear_layers,
        ):
        super().__init__()
        self.encoder = CNN(encoder_channels, linear_layers=[])
        self.decoder = TransposeCNN(decoder_channels, linear_layers=[])

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        h = self.decode(z)
        return h

layers = [784, 256, 64]#, 64, 8, 2]

class AEDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        return (x, x)

    def __len__(self):
        return len(self.dataset)

train_loader = DataLoader(AEDataset(train_data), shuffle=True, batch_size=batch_size)
val_loader = DataLoader(AEDataset(val_data), shuffle=True, batch_size=batch_size)
test_loader = DataLoader(AEDataset(test_data), shuffle=True, batch_size=batch_size)

start_idx = 3
stop_idx = 9
for idx in range(start_idx, stop_idx):
    channels = [
        1, # init channels`
        *[2**idx for idx in range(start_idx, idx+2)]
    ]
    linear_layers = [576, 128]
    print('channels:', channels)
    model = ConvAutoencoder(
        encoder_channels=channels,
        encoder_linear_layers=linear_layers,
        decoder_channels=channels[::-1],
        decoder_linear_layers=linear_layers[::-1]
    )
    print(f'training model {idx}')
    print(model.encoder.layers)
    model, writer = train(
        model=model, 
        logdir='ConvAutoencoder',
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=F.mse_loss,
        epochs=10
    )
    for batch in train_loader:
        x, _ = batch
        reconstructions = model(x)
        x = x.view(x.shape[0], 1, 28, 28)
        reconstructions = reconstructions.view(reconstructions.shape[0], 1, 28, 28)
        visualise_reconstruction(writer, x, reconstructions)
        break