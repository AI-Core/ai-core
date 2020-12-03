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
from ae_utils import visualise_reconstruction
from tuning.tuner import tuner
import json

batch_size = 16

train_data, val_data, test_data = utils.get_splits()

class ConvAutoencoder(torch.nn.Module):
    def __init__(
            self, 
            encoder_channels,
            encoder_linear_layers,
            encoder_kernel_size,
            encoder_stride,
            decoder_channels,
            decoder_linear_layers,
            decoder_kernel_size,
            decoder_stride
        ):
        super().__init__()
        self.encoder = CNN(
            channels=encoder_channels, 
            linear_layers=[],
            kernel_size=encoder_kernel_size,
            stride=encoder_stride
        )
        self.decoder = TransposeCNN(
            channels=decoder_channels, 
            linear_layers=[],
            kernel_size=decoder_kernel_size,
            stride=decoder_stride
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = self.encode(x)
        # print('latent:', x.shape)
        x = self.decode(x)
        return x

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

def on_epoch_end(model, writer, device, epoch):
    model.eval()
    for batch in train_loader:
        x, _ = batch
        x = x.to(device)
        model = model#.to('cpu')
        reconstructions = model(x)
        x = x.view(x.shape[0], 1, 28, 28)
        reconstructions = reconstructions.view(reconstructions.shape[0], 1, 28, 28)
        visualise_reconstruction(writer, x, reconstructions, f'epoch-{epoch}')
        break
    model.train()

def get_channels(strat='paired_exponential'):
    channel_options = []
    if strat == 'exponential':
        start_idx = 6
        stop_idx = 9
        for idx in range(start_idx, stop_idx):
            channels = [
                1, # init channels`
                *[2**idx for idx in range(start_idx, idx+2)]
            ]
            channel_options.append(channels)
    elif strat=='paired_exponential':
        start_idx = 5
        stop_idx = 7
        for idx in range(start_idx, stop_idx):
            c = [1]
            for i in range(start_idx, idx+1):
                val = 2**i
                c.extend([val, val])
            channel_options.append(
                c
            )
    else:
        raise ValueError('Strategy for getting channels not specified')

    return channel_options

def train_tune(config):

        channels = config['channels']
        stride = config['stride']
        kernel_size = config['kernel_size']
   
        linear_layers = [576, 128]
        # print('channels:', channels)
        model = ConvAutoencoder(
            encoder_channels=channels,
            encoder_linear_layers=linear_layers,
            encoder_kernel_size=kernel_size,
            encoder_stride=stride,
            decoder_channels=channels[::-1],
            decoder_linear_layers=linear_layers[::-1],
            decoder_kernel_size=kernel_size,
            decoder_stride=stride
        )
        # print(model.encoder.layers)
        # print(model.decoder.layers)

        # config = {
        #     "lr": tune.qloguniform(1e-4, 1e-1, 0.0001),
        #     # "n_layers": tune.grid_search(list(range(1, 10))),
        #     "batch_size": tune.choice([2, 4, 8, 16])
        # }

        config_str = json.dumps(config)

        if config['optimiser'] == 'sgd':
            optimiser = torch.optim.SGD(model.parameters(), lr=config['lr'])
        elif config['optimiser'] == 'adam':
            optimiser = torch.optim.Adam(model.parameters(), lr=config['lr'])
        else:
            raise ValueError('Optimiser not specified in tuner config')

        model, writer = train(
            model=model,
            optimiser=optimiser,
            logdir='ConvAutoencoder',
            config_str=config_str,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            loss_fn=F.mse_loss,
            epochs=10,
            on_epoch_end=on_epoch_end,
            verbose=False
        )

if __name__ == '__main__':

    print(get_channels())
    # sdf
    tunable_params = {
        'channels': tune.choice(get_channels()),
        'optimiser': tune.choice(['adam', 'sgd']),
        'lr': tune.choice([10**(-idx) for idx in range(1, 5)]),
        'stride': 1,
        'kernel_size': 4
    }

    # TEST FORWARD AND BACKWARD PASSES
    # train_tune(
    #     {
    #         'channels': get_channels()[0],
    #         'optimiser': 'sgd',
    #         'lr': 0.1,
    #         'stride': 1,
    #         'kernel_size': 4
    #     } 
    # )
    
    tuner(train_tune, tunable_params)