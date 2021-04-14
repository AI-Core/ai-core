import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, features_g, z_dim, img_ch, num_classes, embed_size):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.features_g = features_g
        self.noise = nn.Sequential(
            nn.Linear(z_dim, 4 * 4 * features_g * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.embed = nn.Sequential(
            nn.Embedding(num_classes, embed_size),
            nn.Linear(embed_size, 4 * 4),
        )

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # (Channels Z + Embed), 4, 4
            nn.ConvTranspose2d(features_g * 8 + 1, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_g*4) x 8 x 8
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_g*2) x 16 x 16
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_g) x 32 x 32
            nn.ConvTranspose2d(features_g, img_ch, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (img_ch) x 64 x 64
        )


    def forward(self, input, labels):
        noise = self.noise(input).view(input.shape[0], self.features_g * 8, 4, 4)
        embedding = self.embed(labels).view(labels.shape[0], 1, 4, 4)
        input = torch.cat([noise, embedding], dim=1)
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu, features_d, img_ch, num_classes, img_size, embed_size):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.img_size = img_size

        self.embed = nn.Sequential(
            nn.Embedding(num_classes, embed_size),
            nn.Linear(embed_size, img_size * img_size),
        )

        self.main = nn.Sequential(
            # input is (img_ch) x 64 x 64
            nn.Conv2d(img_ch + 1, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d) x 32 x 32
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*2) x 16 x 16
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*4) x 8 x 8
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*8) x 4 x 4
            nn.Conv2d(features_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        input = torch.cat([input, embedding], dim=1)
        return self.main(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)