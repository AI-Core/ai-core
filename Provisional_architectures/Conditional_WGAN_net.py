import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, z_dim, img_ch, features_g, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.main = nn.Sequential(
            self._block(z_dim + embed_size, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, img_ch, 4, 2, 1),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, input, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        input = torch.cat([input, embedding], dim=1)
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, img_ch, features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(img_ch + 1, features_d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(features_d * 8, 1, 4, 2, 0, bias=False),
        )
        self.embed = nn.Embedding(num_classes, img_size*img_size)
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        input = torch.cat([input, embedding], dim=1)
        return self.disc(input)

def gradient_penalty(disc, labels, real, fake, device='cpu'):
    batch_size, ch, h, w = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, ch, h, w).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    mixed_scores = disc(interpolated_images, labels)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_pen = torch.mean((gradient_norm - 1) ** 2)
    return gradient_pen


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)