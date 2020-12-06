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
from ae_utils import visualise_reconstruction, sample
from tuning.tuner import tuner
import json
from architecture import get_channels
import architecture
import random

batch_size = 16

train_data, val_data, test_data = utils.get_splits()

# class AiCoreModel(torch.nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
#         print(kwargs)
#         print('yoooo')
#         sssdfs

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
            decoder_stride,
            decoder_padding,
            unflattened_size,
            verbose=False,
        ):
        self.config = {
            'encoder_channels': encoder_channels,
            'encoder_linear_layers': encoder_linear_layers,
            'encoder_kernel_size': encoder_kernel_size,
            'encoder_stride': encoder_stride,
            'decoder_channels': decoder_channels,
            'decoder_linear_layers': decoder_linear_layers,
            'decoder_kernel_size': decoder_kernel_size,
            'decoder_stride': decoder_stride,
            'decoder_padding': decoder_padding,
            'unflattened_size': unflattened_size
        }
        super().__init__()
        self.encoder = CNN(
            channels=encoder_channels, 
            linear_layers=encoder_linear_layers,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride,
            verbose=verbose,
            activate_last_linear=True
        )
        self.decoder = TransposeCNN(
            channels=decoder_channels, 
            linear_layers=decoder_linear_layers,
            kernel_size=decoder_kernel_size,
            stride=decoder_stride,
            output_padding=decoder_padding,
            unflattened_size=unflattened_size,
            verbose=verbose
        )
        self.verbose = verbose
        print(self.encoder.layers)
        print(self.decoder.layers)

    def set_latent(self, input_size):
        latent_size = architecture.calc_latent_size(input_size, self.config['encoder_channels'], self.config['encoder_kernel_size'], self.config['encoder_stride'])
        self.latent_size = latent_size

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = self.encode(x)
        if self.verbose:
            print('latent:', x.shape)
            print('calculated latent:', self.latent_size)
        x = self.decode(x)
        # scsz
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

def train_tune(config):

        input_size = 28
        ae_arch = architecture.get_ae_architecture(
            input_size=input_size,
            latent_dim=128
        )
        
        model = ConvAutoencoder(**{**ae_arch, 'verbose': True})
        # model = ConvAutoencoder(**ae_arch)

        model.set_latent(input_size)
        # print('model latent dim:', model.latent_size)

        config_str = json.dumps({**config, 'channels': ae_arch['encoder_channels'], 'stride': ae_arch['encoder_stride'], 'kernel_size': ae_arch['encoder_kernel_size'], 'latent_dim': model.latent_size})

        if config['optimiser'] == 'sgd':
            optimiser = torch.optim.SGD(model.parameters(), lr=config['lr'])
        elif config['optimiser'] == 'adam':
            optimiser = torch.optim.Adam(model.parameters(), lr=config['lr'])
        else:
            raise ValueError('Optimiser not specified in tuner config')

        model, writer = train(
            model=model,
            model_class=ConvAutoencoder,
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

    # print(get_channels())

    stride = 2
    kernel_size = 4
    # sdf
    tunable_params = {
        # 'channels': tune.choice(get_channels()),
        'optimiser': tune.choice(['adam', 'sgd']),
        'lr': tune.choice([10**(-idx) for idx in range(1, 5)]),
        # 'stride': tune.choice([2, 3, 4]),
        # 'kernel_size': tune.choice([3, 4, 5])
    }

    # channels = get_channels()[0]

    # TEST FORWARD AND BACKWARD PASSES
    train_tune(
        {
            **tunable_params,
            # 'channels': get_channels()[0],
            'optimiser': 'sgd',
            'lr': 0.1,
            # 'stride': 2,
            # 'kernel_size': 3
        } 
    )
    
    result = tuner(
        train_tune, 
        tunable_params, 
        num_samples=20
    )
            
    best_trial = result.get_best_trial("loss", "min", "last")
    best_checkpoint_dir = best_trial.checkpoint.value
    best_checkpoint_save = os.path.join(best_checkpoint_dir, "checkpoint")
    print(f'best checkpoint found at {best_checkpoint_save}')
    state_dict, model_config, optimiser_state = torch.load(best_checkpoint_save)
    print(model_config)
    best_model = ConvAutoencoder(**model_config)
    best_model.load_state_dict(state_dict)

    sample(best_model.decoder, 32)