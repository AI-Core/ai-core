import torch
import pytorch_lightning as pl
from autoencoders.ConvAutoencoder import ConvAutoencoder

class ConvVAE(ConvAutoencoder):
    def __init__(
        self
        **kwargs
    ):
    super().__init__(**kwargs)
    assert kwargs['encoder_linear_layers'][-1] == 2 * kwargs['decoder_linear_layers'][0]
    self.latent_dim = self.config['decoder_linear_layers']

    def encode(self, x):
        dist_params = self.encoder(x)
        mu_z = dist_params[:self.latent_size]
        mu_s = dist_params[self.latent_size:]

        # REPARAMETERISATION TRICK
        z = mu_z + np.rando.randn(self.latent_size) * mu_s # element wise random sample
        return z


if __name__ == '__main__':
