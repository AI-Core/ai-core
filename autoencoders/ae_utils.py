from torch.utils.tensorboard import SummaryWriter
import torch
from time import time

def visualise_reconstruction(writer, originals, reconstructions, label):

    writer.add_images(f'originals/{label}', originals)
    writer.add_images(f'reconstructions/{label}', reconstructions)

def sample(generator, latent_dim):
    batch_size = 16
    z = torch.randn(batch_size, latent_dim, 1, 1)
    h = generator(z)
    print(h.shape)
    writer = SummaryWriter(log_dir=f'/home/ubuntu/ai-core/runs/Generated-{time()}')
    print('adding imgs')
    writer.add_images(f'Generated/Loss/Gen', h)