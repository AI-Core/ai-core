import torch
import pytorch_lightning as pl
from autoencoders.ConvAutoencoder import ConvAutoencoder


def loss(recon_x, x, mu, logvar):
    # reconstruction losses are summed over all elements and batch
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (recon_loss + kl_diverge) / x.shape[0]  # divide total loss by batch size


class ConvVAE(ConvAutoencoder):
    def __init__(
        self
        **kwargs
    ):
    super().__init__(**kwargs)
    assert kwargs['encoder_linear_layers'][-1] == 2 * kwargs['decoder_linear_layers'][0]
    self.latent_dim = self.config['decoder_linear_layers']
    self.beta = 1

    def encode(self, x):
        dist_params = self.encoder(x) # has size 2*latent_dim
        mu_z = dist_params[:self.latent_size] # first half of the vector is the means of the latent vars, second half is the log of the variances
        logvar_z = dist_params[self.latent_size:] # variance cannot be negative, so assume that this is the log of it

        # REPARAMETERISATION TRICK
        z = mu_z + np.rando.randn(self.latent_size) * logvar_z # element wise random sample
        return z

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encode(x) # the latent representation, which is equivalent to a vector of log probabilities of each latent feature
        pred = self.decode(z)
        loss = F.mse_loss(pred, y)
        loss += self.beta * F.kl_div(z, self.prior(z))
        self.log(self.logdir, loss)

    def prior(self, z):

        # ASSUMING THE PRIOR IS A UNIT GAUSSIAN
        mu = 0
        sigma = 1
        return np.exp(-(z - mu)**2 / (2 * sigma**2) / (sigma * np.sqrt(2 * np.pi))

if __name__ == '__main__':

    
