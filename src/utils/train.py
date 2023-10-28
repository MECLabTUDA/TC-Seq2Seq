import copy

import yaml

import torch
from torch.autograd import Variable
from easydict import EasyDict as edict


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def compute_kl(mu: torch.Tensor):
    mu_squared = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_squared)
    return encoding_loss


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).to(batch.device)
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))  # subtract mean
    return batch


def compute_vgg_loss(vgg, norm, img, target):
    img_vgg = vgg_preprocess(img)
    target_vgg = vgg_preprocess(target)
    img_fea = vgg(img_vgg)
    target_fea = vgg(target_vgg)
    return torch.mean((norm(img_fea) - norm(target_fea)) ** 2)


class DecayLR:

    """ Custom class for decaying learning rate. """

    def __init__(self, epochs, offset, decay_epochs):
        epoch_flag = epochs - decay_epochs
        assert (epoch_flag > 0), "Decay must start before the training session ends!"
        self.epochs = epochs
        self.offset = offset
        self.decay_epochs = decay_epochs

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epochs) / (
                self.epochs - self.decay_epochs)


def read_config(path: str) -> (dict, dict):

    with open(path, 'r') as conf_file:
        conf_dict = yaml.safe_load(conf_file)

    return edict(copy.deepcopy(conf_dict)), conf_dict
