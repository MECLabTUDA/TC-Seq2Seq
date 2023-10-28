import unittest

import torch

from src.model import CycleGAN_Generator, CycleGAN_Discriminator
from src.utils.train import read_config

DEVICE = 'cuda'  # 'cuda:2'


class PreTrainTests(unittest.TestCase):

    def setUp(self) -> None:
        self.img_dim = 128
        self.gen = CycleGAN_Generator(img_dim=self.img_dim).to(DEVICE)
        self.disc = CycleGAN_Discriminator().to(DEVICE)

    def test_read_conf(self):
        path = 'src/config/cyclegan_cadis_cataract.yml'
        conf = read_config(path)[0]
        self.assertTrue(conf.data is not None)

    def test_gen_output_shape(self):
        """ Checks the generator model to generate tensors of the input dimension. """
        fake_img = torch.rand(size=(1, 3, self.img_dim, self.img_dim)).to(DEVICE)
        gen_pred = self.gen(fake_img)
        self.assertEqual(fake_img.shape, gen_pred.shape)

    def test_disc_output_shape(self):
        """ Checks the discriminator model to output a single numerical prediction. """
        fake_img = torch.rand(size=(1, 3, self.img_dim, self.img_dim)).to(DEVICE)
        disc_pred = self.disc(fake_img)
        self.assertEqual(disc_pred.shape, torch.Size([1, 1]))


if __name__ == "__main__":
    unittest.main()
