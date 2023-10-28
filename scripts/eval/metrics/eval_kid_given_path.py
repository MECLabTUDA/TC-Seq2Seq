import os
import sys

import argparse
import torch
from natsort import natsorted
from torchvision.io import read_image
from torchmetrics.image.kid import KernelInceptionDistance

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='Data root path')
parser.add_argument('--id', type=str, help='id of eval. run', default="")
parser.add_argument('--dev', type=str, help='device', default='cpu')
ops = parser.parse_args()

print(ops.id)
#
# B -> A
#

kid = KernelInceptionDistance(subset_size=50).to(ops.dev)
real_seq_paths = [ops.path + "test_samples/real_A/" + sample + "/"
                  for sample in natsorted(os.listdir(ops.path + "test_samples/real_A/"))]
fake_seq_paths = [ops.path + "test_samples/fake_BA/" + sample + "/"
                  for sample in natsorted(os.listdir(ops.path + "test_samples/fake_BA/"))]

for (real_seq_path, fake_seq_path) in zip(real_seq_paths, fake_seq_paths):
    real_seq = torch.cat([read_image(real_seq_path + path).unsqueeze(0)
                          for path in natsorted(os.listdir(real_seq_path))]).to(ops.dev)
    fake_seq = torch.cat([read_image(fake_seq_path + path).unsqueeze(0)
                          for path in natsorted(os.listdir(fake_seq_path))]).to(ops.dev)
    kid.update(real_seq, real=True)
    kid.update(fake_seq, real=False)

kid_mean, kid_std = kid.compute()
print("KID(BA, A): ", (kid_mean, kid_std))

#
# B -> A
#

kid = KernelInceptionDistance(subset_size=50).to(ops.dev)
real_seq_paths = [ops.path + "test_samples/real_B/" + sample + "/"
                  for sample in natsorted(os.listdir(ops.path + "test_samples/real_B/"))]
fake_seq_paths = [ops.path + "test_samples/fake_AB/" + sample + "/"
                  for sample in natsorted(os.listdir(ops.path + "test_samples/fake_AB/"))]

for (real_seq_path, fake_seq_path) in zip(real_seq_paths, fake_seq_paths):
    real_seq = torch.cat([read_image(real_seq_path + path).unsqueeze(0)
                          for path in natsorted(os.listdir(real_seq_path))]).to(ops.dev)
    fake_seq = torch.cat([read_image(fake_seq_path + path).unsqueeze(0)
                          for path in natsorted(os.listdir(fake_seq_path))]).to(ops.dev)
    kid.update(real_seq, real=True)
    kid.update(fake_seq, real=False)

kid_mean, kid_std = kid.compute()
print("KID(AB, B): ", (kid_mean, kid_std))
