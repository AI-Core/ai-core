from torch.utils.tensorboard import SummaryWriter
import torch

def visualise_reconstruction(writer, originals, reconstructions, label):

    writer.add_images(f'originals/{label}', originals)
    writer.add_images(f'reconstructions/{label}', reconstructions)

def sample(generator, latent_dim):
    batch_size = 16
    z = torch.randn(batch_size, latent_dim, 1, 1)
    h = generator(z)
    writer = SummaryWriter(log_dir='/home/ubuntu/ai-core/runs/Generated')
    print('adding imgs')
    writer.add_images('generated', h)