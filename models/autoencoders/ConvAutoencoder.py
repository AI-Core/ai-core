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
from time import time

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from pytorch_lightning.callbacks import Callback
from ReconstructionDataset import ReconstructionDataset
from autoencoders.callbacks import SampleReconstructionCallback

class ConvAutoencoder(pl.LightningModule):
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
            optimizer_name=None,
            lr=None,
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
            'unflattened_size': unflattened_size,
            'optimizer_name': optimizer_name,
            'lr': lr
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
        # latent_size = architecture.calc_latent_size(input_size, self.config['encoder_channels'], self.config['encoder_kernel_size'], self.config['encoder_stride']) # if it has no linear layers
        latent_size = self.config['decoder_linear_layers'][0] # if it has linear layers in the middle (had to use the decoder rather than encoder becuase for VAE encoder last layer=2*latent_dim)
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.decoder(self.encoder(x))
        loss = F.mse_loss(pred, y)
        self.log(self.logdir, loss)
        return loss

    def validation_step(self, *args):
        loss = self.training_step(*args)
        return {'val_loss': loss}

    def configure_optimizers(self):

        if self.config['optimizer_name'] == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config['lr'])
        elif self.config['optimizer_name'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        else:
            raise ValueError('Optimizer not specified in tuner config')

        # optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

def trainable(config, train_loader, val_loader, test_loader):

    input_size = 28
    ae_arch = architecture.get_ae_architecture(
        input_size=input_size,
        latent_dim=128
    )
    
    # model = ConvAutoencoder(**{**ae_arch, 'verbose': True})
    model = ConvAutoencoder(**{**ae_arch, 'optimizer_name': config['optimizer_name'], 'lr': config['lr']})
    model.logdir = 'ConvAutoencoder'

    model.set_latent(input_size)
    # print('model latent dim:', model.latent_size)

    config_str = json.dumps({**config, 'channels': ae_arch['encoder_channels'], 'stride': ae_arch['encoder_stride'], 'kernel_size': ae_arch['encoder_kernel_size'], 'latent_dim': model.latent_size})

    # SET UP LOGGER
    section_name = 'ConvAutoencoder'
    save_dir =f'{os.path.expanduser("~")}/ai-core/Embedder/runs/{section_name}/'
    # save_dir =f'{os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")}/runs/{section_name}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # print(save_dir)
    # print(__name__)
    # print(__file__)
    # sdfcds
    experiment_name = f'ConvAutoencoder-{config_str}-{time()}'
    model.experiment_name = experiment_name
    logger = pl.loggers.TensorBoardLogger(
        save_dir=save_dir,
        name=experiment_name,
        default_hp_metric=False,
    )

    # CREATE CHECKPOINTS DIR
    checkpoint_dir = f'checkpoints/{experiment_name}'
    os.makedirs(checkpoint_dir)

    # RUN TRAINER
    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=1,
        max_epochs=10,
        val_check_interval=0.05, # for dev
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                metrics={"loss": "val_loss",},
                on="validation_end"
            ),
            TuneReportCheckpointCallback(
                metrics={"loss": "val_loss"},
                filename=f"{checkpoint_dir}/latest_checkpoint.ckpt", # TODO edit callback so that it saves history of checkpoints and make PR to ray[tune]
                on="validation_end"
            ),
            SampleReconstructionCallback(loader=val_loader)
        ]
    )
    trainer.fit(model, 
        train_dataloader=train_loader,
        val_dataloaders=val_loader
    )
    test_result = Trainer.test(
        model=model,
        test_dataloaders=test_loader,
        verbose=True
    )
    # model, writer = train(
    #     model=model,
    #     model_class=ConvAutoencoder,
    #     optimiser=optimiser,
    #     logdir='ConvAutoencoder',
    #     config_str=config_str,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     test_loader=test_loader,
    #     loss_fn=F.mse_loss,
    #     epochs=10,
    #     on_epoch_end=on_epoch_end,
    #     verbose=False
    # )

if __name__ == '__main__':

    # print(get_channels())
    batch_size = 16

    train_data, val_data, test_data = utils.get_splits()
    train_loader = DataLoader(ReconstructionDataset(train_data), shuffle=True, batch_size=batch_size, num_workers=8)
    val_loader = DataLoader(ReconstructionDataset(val_data), shuffle=True, batch_size=batch_size, num_workers=8)
    test_loader = DataLoader(ReconstructionDataset(test_data), shuffle=True, batch_size=batch_size)


    stride = 2
    kernel_size = 4
    # sdf
    tunable_params = {
        # 'channels': tune.choice(get_channels()),
        'optimizer_name': tune.choice(['adam', 'sgd']),
        'lr': tune.choice([10**(-idx) for idx in range(1, 5)]),
        # 'stride': tune.choice([2, 3, 4]),
        # 'kernel_size': tune.choice([3, 4, 5])
    }

    # channels = get_channels()[0]

    # TEST FORWARD AND BACKWARD PASSES
    # trainable(
    #     {
    #         **tunable_params,
    #         # 'channels': get_channels()[0],
    #         # 'optimiser': 'sgd',
    #         # 'lr': 0.1,
    #         # 'stride': 2,
    #         # 'kernel_size': 3
    #     } 
    # )
    # dsds
    result = tuner(
        tune.with_parameters(
            trainable,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        ),
        tunable_params, 
        num_samples=1
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