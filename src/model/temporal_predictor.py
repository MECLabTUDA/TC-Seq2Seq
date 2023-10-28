import torch.nn.functional as F
import torch.nn as nn
import torch

"""
    Temporal predictor model, following https://github.com/SunnerLi/RecycleGAN and 
    https://github.com/milesial/Pytorch-UNet
    
"""


class TemporalPredictorModel(nn.Module):
    def __init__(self, conf):
        """ Creates class instance

        :param conf: Configuration dictionary
        """
        super().__init__()
        n_in = conf.n_in * conf.nt
        n_out = conf.n_out
        r = conf.r
        self.inc = InputConvolution(n_in, 64 // r)
        self.down1 = DownsampleBlock(64 // r, 128 // r)
        self.down2 = DownsampleBlock(128 // r, 256 // r)
        self.down3 = DownsampleBlock(256 // r, 512 // r)
        self.down4 = DownsampleBlock(512 // r, 512 // r)
        self.up1 = UpsampleBlock(1024 // r, 256 // r, bilinear=True)
        self.up2 = UpsampleBlock(512 // r, 128 // r, bilinear=True)
        self.up3 = UpsampleBlock(256 // r, 64 // r, bilinear=True)
        self.up4 = UpsampleBlock(128 // r, 64 // r, bilinear=True)
        self.outc = OutputConvolution(64 // r, n_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = torch.tanh(x)
        return x


class DoubleConvBlock(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InputConvolution(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InputConvolution, self).__init__()
        self.conv = DoubleConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownsampleBlock, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(UpsampleBlock, self).__init__()
        self.bilinear = bilinear
        if not bilinear:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = DoubleConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutputConvolution(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutputConvolution, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
