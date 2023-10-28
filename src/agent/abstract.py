import torch


class Agent:

    """ Abstract agent class for models. """

    def __init__(self, conf: dict):
        pass

    def get_opt_and_scheduler(self, conf: dict):
        raise NotImplementedError


class GAN_Agent(Agent):

    def __init__(self, conf: dict):
        super().__init__(conf)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def discriminator_update(self):
        raise NotImplementedError

    def generator_update(self):
        raise NotImplementedError
