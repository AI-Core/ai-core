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
from get_channels import get_channels

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
            verbose=False
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
            'decoder_padding': decoder_padding
        }
        super().__init__()
        self.encoder = CNN(
            channels=encoder_channels, 
            linear_layers=encoder_linear_layers,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride,
            verbose=verbose
        )
        self.decoder = TransposeCNN(
            channels=decoder_channels, 
            linear_layers=decoder_linear_layers,
            kernel_size=decoder_kernel_size,
            stride=decoder_stride,
            output_padding=decoder_padding,
            verbose=verbose
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = self.encode(x)
        # print('latent:', x.shape)
        # scsds
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

def train_tune(config):


        channels = config['channels']
        stride = config['stride']
        kernel_size = config['kernel_size']

        channel_sizes, remainders = utils.calc_channel_size(28, channels, kernel_size, stride)
        utils.calc_transpose_channel_size(channel_sizes[-1], channels[::-1], kernel_size, stride, remainders)
        output_padding = remainders[::-1] # need to reverse to mirror order of layers and apply matching

        linear_layers = [1, 128]
        linear_layers = []
        # print('channels:', channels)
        model = ConvAutoencoder(
            encoder_channels=channels,
            encoder_linear_layers=linear_layers,
            encoder_kernel_size=kernel_size,
            encoder_stride=stride,
            decoder_channels=channels[::-1],
            decoder_linear_layers=linear_layers[::-1],
            decoder_kernel_size=kernel_size,
            decoder_stride=stride,
            decoder_padding=output_padding,
            verbose=False
        )
        print(model.encoder.layers)
        print(model.decoder.layers)

        latent_size = utils.calc_latent_size(28, channels, kernel_size, stride)
        model.latent_size = latent_size


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
            model_class=ConvAutoencoder,
            optimiser=optimiser,
            logdir='ConvAutoencoder',
            config_str=config_str,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            loss_fn=F.mse_loss,
            epochs=1,
            on_epoch_end=on_epoch_end,
            verbose=False
        )

if __name__ == '__main__':

    print(get_channels())

    stride = 2
    kernel_size = 4
    # sdf
    tunable_params = {
        'channels': tune.choice(get_channels()),
        'optimiser': tune.choice(['adam', 'sgd']),
        'lr': tune.choice([10**(-idx) for idx in range(1, 5)]),
        'stride': tune.choice([2, 3, 4]),
        'kernel_size': tune.choice([3, 4, 5])
    }

    # channels = get_channels()[0]

    # TEST FORWARD AND BACKWARD PASSES
    # train_tune(
    #     {
    #         **tunable_params,
    #         'channels': get_channels()[0],
    #         'optimiser': 'sgd',
    #         'lr': 0.1,
    #     } 
    # )
    
    result = tuner(
        train_tune, 
        tunable_params, 
        num_samples=2
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