import torch
import torch.nn as nn
from torch.autograd import Variable

from .layers import ResidualBlock
from .UNIT_autoencoder_v1 import ContentEncoder, Decoder, StyleEncoder


class CycleGAN_Generator(nn.Module):
    def __init__(self, img_dim: int, residual_block_dim: int = 256):
        super(CycleGAN_Generator, self).__init__()

        if img_dim <= 128:
            self.num_res_blocks = 6
        else:
            self.num_res_blocks = 9

        seq_model = []
        # Initial convolution block; c7s1-64
        seq_model.extend([
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ])
        dim = 64
        while dim < residual_block_dim:
            # Downsampling  dXXX
            seq_model.extend([
                nn.Conv2d(dim, dim*2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(dim*2),
                nn.ReLU(inplace=True)
            ])
            dim *= 2

        # Residual blocks
        seq_model.extend(nn.Sequential(*[ResidualBlock(residual_block_dim) for _ in range(self.num_res_blocks)]))

        while dim > 64:
            seq_model.extend([
                # Upsampling
                nn.ConvTranspose2d(dim, dim//2, 3, stride=2, padding=1, output_padding=1),  # u128
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True),
            ])
            dim //= 2

        seq_model.extend([
            # Output layer
            nn.ReflectionPad2d(3),  # c7s1-3
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        ])

        self.main = nn.Sequential(*seq_model)

    def forward(self, x):
        return self.main(x)


class UNIT_VAEGen(nn.Module):
    # VAE architecture
    def __init__(self, input_dim, params):
        super(UNIT_VAEGen, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activation']
        pad_type = params['padding_type']

        # content encoder
        self.enc = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim,
                           res_norm='in', activ=activ, pad_type=pad_type)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are
        # multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training == True:
            # noise = Variable(torch.randn(hiddens.size()).to(hiddens.data.get_device()))
            noise = Variable(torch.randn(hiddens.size()).to(images.device))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()).to(images.device))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images
