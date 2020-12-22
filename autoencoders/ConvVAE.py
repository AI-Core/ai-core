import torch
import pytorch_lightning as pl
from ConvAutoencoder import ConvAutoencoder
import numpy as np
import torch.nn.functional as F

def dkl(mu, logvar):

    kl_diverge = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_diverge


class ConvVAE(ConvAutoencoder):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert kwargs['encoder_linear_layers'][-1] == 2 * kwargs['decoder_linear_layers'][0]
        self.latent_size = self.config['decoder_linear_layers'][0]
        self.beta = 1
        self.prior = Gaussian()

    def encode(self, x):
        dist_params = self.encoder(x) # has size 2*latent_dim
        mu_z = dist_params[:, :self.latent_size] # first half of the vector is the means of the latent vars, second half is the log of the variances
        logvar_z = dist_params[:, self.latent_size:] # variance cannot be negative, so assume that this is the log of it which can be negative
        # print('dist params shape:', dist_params.shape)
        # print('latent size:', self.latent_size)
        # REPARAMETERISATION TRICK
        # print('epsilon shape:', torch.randn_like(mu_z).shape)
        # print('logvar shape:', logvar_z.shape)
        z = mu_z + torch.randn_like(mu_z) * logvar_z # element wise random sample


        Q = Gaussian(mu=mu_z, sigma=torch.exp(logvar_z))
        print()
        z = torch.zeros_like(z)
        print('z')
        print(z)
        print()
        print('Q(z):', Q(z))
        print()
        print('log(Q(z)):', torch.log(Q(z)))
        print()
        # print('prior:', self.prior(z))

        kl_loss = F.kl_div(torch.log(Q(z)), self.prior(z))
        print('pt kl_loss:', kl_loss)
        print('my kl_loss:', dkl(mu_z, logvar_z))
        print(batch_size*dkl(mu_z, logvar_z))
        assert kl_loss == dkl(mu_z, logvar_z)

        return z

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encode(x) # the latent representation, which is equivalent to a vector of log probabilities of each latent feature
        pred = self.decode(z)
        loss = F.mse_loss(pred, y)
        loss += self.beta * F.kl_div(z, self.prior(z))
        self.log(self.logdir, loss)
        return loss

class Gaussian():
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(
            - (x - self.mu)**2
            /
            (2 * self.sigma**2)
        ) / (self.sigma * np.sqrt(2 * np.pi))

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from architecture import get_vae_architecture
    from ReconstructionDataset import ReconstructionDataset
    from callbacks import SampleReconstructionCallback
    import utils
    import os
    import json
    from time import time
    from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

    batch_size = 2
    train_data, val_data, test_data = utils.get_splits()
    train_loader = DataLoader(ReconstructionDataset(train_data), shuffle=True, batch_size=batch_size, num_workers=8)
    val_loader = DataLoader(ReconstructionDataset(val_data), shuffle=True, batch_size=batch_size, num_workers=8)
    test_loader = DataLoader(ReconstructionDataset(test_data), shuffle=True, batch_size=batch_size)

    input_size=28
    vae_arch = get_vae_architecture(
        input_size=input_size,
        latent_dim=4
    )

    model = ConvVAE(**{**vae_arch, 'optimizer_name': 'adam', 'lr': 0.0001})
    model.logdir = 'VAE'

    config_str = json.dumps({'encoder_channels': vae_arch['encoder_channels'], 'optimizer_name': 'adam', 'lr': 0.0001})

    # SET UP LOGGER
    section_name = 'VAE'
    save_dir =f'{os.path.expanduser("~")}/ai-core/Embedder/runs/{section_name}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    experiment_name = f'{section_name}-{config_str}-{time()}'
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
        progress_bar_refresh_rate=1,
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
