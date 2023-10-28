import itertools

import torch.optim as optim

from src.model import CycleGAN_Generator, CycleGAN_Discriminator
from src.model.temporal_predictor import TemporalPredictorModel
from src.utils.model import init_weights
from src.utils.train import DecayLR

from .abstract import GAN_Agent


class ReCycleGAN(GAN_Agent):

    def __init__(self, conf: dict):
        super(ReCycleGAN, self).__init__(conf)

        self.netG_A2B = CycleGAN_Generator(conf.data.img_dim, conf.model.latent_dim).to(conf.device)
        self.netG_B2A = CycleGAN_Generator(conf.data.img_dim, conf.model.latent_dim).to(conf.device)
        self.netD_A = CycleGAN_Discriminator().to(conf.device)
        self.netD_B = CycleGAN_Discriminator().to(conf.device)
        self.temp_pred_A = TemporalPredictorModel(conf.model.temp_pred).to(conf.device)
        self.temp_pred_B = TemporalPredictorModel(conf.model.temp_pred).to(conf.device)

        self.netG_A2B.apply(lambda m: init_weights(m, conf.training.weight_init))
        self.netG_B2A.apply(lambda m: init_weights(m, conf.training.weight_init))
        self.netD_A.apply(lambda m: init_weights(m, conf.training.weight_init))
        self.netD_B.apply(lambda m: init_weights(m, conf.training.weight_init))
        self.temp_pred_A.apply(lambda m: init_weights(m, conf.training.weight_init))
        self.temp_pred_B.apply(lambda m: init_weights(m, conf.training.weight_init))

        self.optimizer_G = None
        self.optimizer_D_A = None
        self.optimizer_D_B = None
        self.lr_scheduler_G = None
        self.lr_scheduler_D_A = None
        self.lr_scheduler_D_B = None

    def get_opt_and_scheduler(self, conf: dict):
        self.optimizer_G = optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters(),
                                                      self.temp_pred_A.parameters(), self.temp_pred_B.parameters()),
                                      lr=conf.training.initial_lr, betas=eval(conf.training.adam_betas))
        self.optimizer_D_A = optim.Adam(self.netD_A.parameters(), lr=conf.training.initial_lr,
                                        betas=eval(conf.training.adam_betas))
        self.optimizer_D_B = optim.Adam(self.netD_B.parameters(), lr=conf.training.initial_lr,
                                        betas=eval(conf.training.adam_betas))

        # Learning rate schedulers
        lr_lambda = DecayLR(epochs=conf.training.epochs, offset=0, decay_epochs=conf.training.lr_decay_start_epoch).step
        self.lr_scheduler_G = optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lr_lambda)
        self.lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=lr_lambda)
        self.lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=lr_lambda)


