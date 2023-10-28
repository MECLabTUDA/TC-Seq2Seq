import random
import math

import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg16_bn, VGG16_BN_Weights


def load_vgg16(model_dir):
    m = vgg16_bn()
    return m

# LOL

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def init_weights(m: torch.nn.Module, method: str):
    # if type(m) in [nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm1d, nn.Linear]:
    if type(m) in [nn.Conv2d, nn.Linear]:
        if method == 'kaiming uniform':
            nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif method == 'xavier uniform':
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif method == 'kaiming normal':
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.bias.data.fill_(0.01)
        elif method == 'xavier normal':
            nn.init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            m.bias.data.fill_(0.01)


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert (max_size > 0), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


class ApplyNoise(nn.Module):
    r"""Add Gaussian noise to the input tensor."""

    def __init__(self):
        super().__init__()
        # scale of the noise
        self.scale = nn.Parameter(torch.zeros(1))
        self.conditional = True

    def forward(self, x, *_args, noise=None, **_kwargs):
        r"""
        Args:
            x (tensor): Input tensor.
            noise (tensor, optional, default=``None``) : Noise tensor to be
                added to the input.
        """
        if noise is None:
            sz = x.size()
            noise = x.new_empty(sz[0], 1, *sz[2:]).normal_()

        return x + self.scale * noise
