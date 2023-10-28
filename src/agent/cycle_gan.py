import itertools

import torch.optim as optim

from src.model import CycleGAN_Generator, CycleGAN_Discriminator
from src.utils.model import weights_init
from src.utils.train import DecayLR

from .abstract import GAN_Agent


class CycleGAN_Agent(GAN_Agent):

    def __init__(self, conf: dict):
        super(CycleGAN_Agent, self).__init__(conf)
        self.netG_A2B = CycleGAN_Generator(conf.data.img_dim, conf.model.latent_dim).to(conf.device)
        self.netG_B2A = CycleGAN_Generator(conf.data.img_dim, conf.model.latent_dim).to(conf.device)
        self.netD_A = CycleGAN_Discriminator().to(conf.device)
        self.netD_B = CycleGAN_Discriminator().to(conf.device)

        self.netG_A2B.apply(weights_init)
        self.netG_B2A.apply(weights_init)
        self.netD_A.apply(weights_init)
        self.netD_B.apply(weights_init)

        self.optimizer_G = None
        self.optimizer_D_A = None
        self.optimizer_D_B = None
        self.lr_scheduler_G = None
        self.lr_scheduler_D_A = None
        self.lr_scheduler_D_B = None

    def get_opt_and_scheduler(self, conf: dict):
        self.optimizer_G = optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                      lr=conf.training.initial_lr,
                                      betas=eval(conf.training.adam_betas),
                                      weight_decay=eval(conf.training.weight_decay))
        self.optimizer_D_A = optim.Adam(self.netD_A.parameters(),
                                        lr=conf.training.initial_lr,
                                        betas=eval(conf.training.adam_betas),
                                        weight_decay=eval(conf.training.weight_decay))
        self.optimizer_D_B = optim.Adam(self.netD_B.parameters(),
                                        lr=conf.training.initial_lr,
                                        betas=eval(conf.training.adam_betas),
                                        weight_decay=eval(conf.training.weight_decay))

        # Learning rate schedulers
        lr_lambda = DecayLR(epochs=conf.training.epochs, offset=0, decay_epochs=conf.training.lr_decay_start_epoch).step
        self.lr_scheduler_G = optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lr_lambda)
        self.lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=lr_lambda)
        self.lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=lr_lambda)

