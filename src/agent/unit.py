import os.path

import torch
import torch.nn as nn

from src.model.discriminator import UNIT_MsImageDis
from src.model.generator import UNIT_VAEGen
from src.utils.model import load_vgg16, init_weights
from src.utils.train import DecayLR

from .abstract import GAN_Agent


class UNIT(GAN_Agent):

    def __init__(self, conf: dict):
        super(UNIT, self).__init__(conf)
        self.gen_A = UNIT_VAEGen(conf.data.in_dim_A, conf.model.gen).to(conf.device)
        self.gen_B = UNIT_VAEGen(conf.data.in_dim_B, conf.model.gen).to(conf.device)
        self.disc_A = UNIT_MsImageDis(conf.data.in_dim_A, conf.model.disc).to(conf.device)
        self.disc_B = UNIT_MsImageDis(conf.data.in_dim_B, conf.model.disc).to(conf.device)
        self.instance_norm = nn.InstanceNorm1d(512, affine=False).to(conf.device)

        # Weight init
        self.gen_A.apply(lambda m: init_weights(m, conf.training.weight_init))
        self.gen_B.apply(lambda m: init_weights(m, conf.training.weight_init))
        self.disc_A.apply(lambda m: init_weights(m, conf.training.weight_init))
        self.disc_B.apply(lambda m: init_weights(m, conf.training.weight_init))

        if conf.model.vgg_weight > 0.0:
            self.vgg = load_vgg16(conf.model.vgg_model_path + '/models').to(conf.device)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def load_from_checkpoint(self, chckpt_path: str):
        assert os.path.isdir(chckpt_path)
        self.gen_A.load_state_dict(torch.load(chckpt_path + 'gen_A_epoch139.pth', map_location='cpu'))
        self.gen_B.load_state_dict(torch.load(chckpt_path + 'gen_B_epoch139.pth', map_location='cpu'))
        self.disc_A.load_state_dict(torch.load(chckpt_path + 'disc_A_epoch139.pth', map_location='cpu'))
        self.disc_B.load_state_dict(torch.load(chckpt_path + 'disc_B_epoch139.pth', map_location='cpu'))
        self.disc_opt.load_state_dict(torch.load(chckpt_path + 'disc_opt_epoch139.pth', map_location='cpu'))
        self.gen_opt.load_state_dict(torch.load(chckpt_path + 'gen_opt_epoch139.pth', map_location='cpu'))

    def get_opt_and_scheduler(self, conf: dict):
        disc_params = list(self.disc_A.parameters()) + list(self.disc_B.parameters())
        self.disc_opt = torch.optim.Adam(
            [p for p in disc_params if p.requires_grad],
            lr=conf.training.initial_lr,
            betas=eval(conf.training.adam_betas),
            weight_decay=eval(conf.training.weight_decay)
        )

        gen_params = list(self.gen_A.parameters()) + list(self.gen_B.parameters())
        self.gen_opt = torch.optim.Adam(
            [p for p in gen_params if p.requires_grad],
            lr=conf.training.initial_lr,
            betas=eval(conf.training.adam_betas),
            weight_decay=eval(conf.training.weight_decay)
        )

        lr_lambda = DecayLR(epochs=conf.training.epochs, offset=0, decay_epochs=conf.training.lr_decay_start_epoch).step
        self.lr_scheduler_disc = torch.optim.lr_scheduler.LambdaLR(self.disc_opt, lr_lambda=lr_lambda)
        self.lr_scheduler_gen = torch.optim.lr_scheduler.LambdaLR(self.gen_opt, lr_lambda=lr_lambda)
