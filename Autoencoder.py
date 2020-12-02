import torch
import utils
from NN import NN 
import numpy as np
import torch.nn.functional as F
from ray import tune
import os
from train import train
from torch.utils.data import DataLoader

batch_size = 16

train_data, val_data, test_data = utils.get_splits()

class AutoEncoder(torch.nn.Module):
    def __init__(
            self, 
            encoder_layers, 
            decoder_layers
        ):
        super().__init__()
        self.encoder = NN(encoder_layers)
        self.decoder = NN(decoder_layers)

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

def flatten(x):
    return torch.flatten(x)

train_loader = DataLoader(AEDataset(train_data, transform=flatten), shuffle=True, batch_size=batch_size)
val_loader = DataLoader(AEDataset(val_data, transform=flatten), shuffle=True, batch_size=batch_size)
test_loader = DataLoader(AEDataset(test_data, transform=flatten), shuffle=True, batch_size=batch_size)

class FlatAE(torch.nn.Module):
    def __init__(self,
        encoder_layers,
        decoder_layers
    ):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            AutoEncoder(
                encoder_layers,
                decoder_layers
            )
        )

    def forward(self, x):
        return self.layers(x)

# ae = FlatAE()

def sample(writer, device='cpu'):
    z = torch.randn(batch_size, latent_vec_size).to(device)
    for img in G(z):
        writer.add_image(f'test', img)

def visualise_reconstruction(writer, originals, reconstructions):

    writer.add_images('originals', originals)
    writer.add_images('reconstructions', reconstructions)

for idx in range(9):
    layers = [
        784,
        *[2**(9-idx) for idx in range(9)]
    ]
    print(layers)
    model = FlatAE(
        encoder_layers=layers,
        decoder_layers=layers[::-1]
    )
    print(f'training model {idx}')
    model, writer = train(
        model=model, 
        logdir='Autoencoder',
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