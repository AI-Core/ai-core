import torch.nn as nn
import torch
from tqdm import tqdm


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


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


class CWGAN:
    def __init__(self, ngpu=1, workers=2, learning_rate=1e-4, optim=None, beta_1=0.0, beta_2=0.9,
                 image_size=64, image_channels=3, gen_embedding=100, z_dim=100, features_g=64, features_d=64,
                 critic_iterations=5, lambda_penalty=10):
        print('This model was provided by Ivan Ying Xuan and Gameli Ladzekpo')
        self.ngpu = ngpu
        self.workers = workers
        self.learning_rate = learning_rate
        self.optim = optim
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.image_size = image_size
        self.image_channels = image_channels
        self.gen_embedding = gen_embedding
        self.z_dim = z_dim
        self.features_g = features_g
        self.features_d = features_d
        self.critic_iteration=critic_iterations
        self.lambda_penalty = lambda_penalty
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    def fit(self, dataset, epochs, batch_size, save_model=True):
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=self.workers)
        netD = Discriminator(
            self.image_channels,
            self.features_d,
            dataset.num_classes,
            self.image_size)
        netG = Generator(
            self.z_dim,
            self.image_channels,
            self.features_g,
            dataset.num_classes,
            self.image_size
        )
        # Handle multi-gpu if desired
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            netD = nn.DataParallel(netD, list(range(self.ngpu)))
            netG = nn.DataParallel(netG, list(range(self.ngpu)))


        netD.apply(weights_init)
        netG.apply(weights_init)

        if self.optim is None:
            self.optim = torch.optim.Adam()
        # Setup Adam optimizers for both G and D
        optimizerD = self.optim(
            netD.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta_2))
        optimizerG = self.optim(
            netG.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta_2))

        # fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)
        # os.makedirs('model/cWGAN', exist_ok=True)
        # with open('model/cWGAN/image_data.json', 'w') as f:
        #     json.dump(dataset_info, f)
            
        # # Number of training epochs
        # writer_real = SummaryWriter(log_dir=f'logs_runs/logs_cWGAN/real')
        # writer_fake = SummaryWriter(log_dir=f'logs_runs/logs_cWGAN/fake')
        # writer_lossD = SummaryWriter(log_dir=f'logs_runs/logs_cWGAN/lossD')
        # writer_lossG = SummaryWriter(log_dir=f'logs_runs/logs_cWGAN/lossG')
        # writer_penalty = SummaryWriter(log_dir=f'logs_runs/logs_cWGAN/penalty')
        # iters = 0
        # suffix = f'lr={learning_rate}_beta={beta1}_batch={batch_size}'  
        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(epochs):
            print(f"Epoch [{epoch+1}/{epochs}]")
            for i, (real, labels) in enumerate(tqdm(dataloader)):
                real = real.to(self.device)
                cur_batch_size = real.shape[0]
                labels = labels.to(self.device)
                for _ in range(self.critic_iterations):
                    noise = torch.randn(cur_batch_size, self.z_dim, 1, 1).to(self.device)
                    fake = netG(noise, labels)
                    disc_real = netD(real, labels).reshape(-1)
                    disc_fake = netD(fake, labels).reshape(-1)
                    gp = gradient_penalty(netD, labels, real, fake, device=self.device)
                    errD_real = torch.mean(disc_real)
                    errD_fake = torch.mean(disc_fake)
                    loss_disc = (-(errD_real - errD_fake)\
                                + self.lambda_penalty * gp)
                    netD.zero_grad()
                    loss_disc.backward(retain_graph=True)
                    optimizerD.step()

                gen_fake = netD(fake, labels).view(-1)
                loss_gen = -torch.mean(gen_fake)
                netG.zero_grad()
                loss_gen.backward()
                optimizerG.step()

                # Output training stats
                # if iters % 25 == 0:
                #     print(
                #         f'Epoch [{epoch}/{epochs}] Batch {i}/{len(dataloader)} '
                #         + f'Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}'
                #         )
                #     with torch.no_grad():
                #         fake = netG(noise, labels)
                #         img_grid_real = torchvision.utils.make_grid(
                #             real[:32], normalize=True
                #         )
                #         img_grid_fake = torchvision.utils.make_grid(
                #             fake[:32], normalize=True
                #         )

                #         writer_real.add_image('Real', img_grid_real, global_step=iters)
                #         writer_real.add_scalar('D(x)', errD_real, global_step=iters)
                #         writer_fake.add_image('Fake', img_grid_fake, global_step=iters)
                #         writer_fake.add_scalar('D(G(z))', errD_fake, global_step=iters)
                #         writer_lossD.add_scalar('Loss_Discriminator', loss_disc.item(), global_step=iters)
                #         writer_lossG.add_scalar('Loss_Generator', loss_gen.item(), global_step=iters)
                #         writer_penalty.add_scalar('Gradient_Penalty', gp.item(), global_step=iters)
                # iters += 1
            if save_model:
                save_checkpoint(netG, optimizerG, filename="generator.pt")
                save_checkpoint(netD, optimizerD, filename="discriminator.pt")
        # torch.save(netD, 'model/cWGAN/Discriminator.pt')
        # torch.save(netG, 'model/cWGAN/Generator.pt')