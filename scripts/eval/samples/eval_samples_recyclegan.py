import os
import random
import shutil
import time

import torch
import numpy as np
from tqdm import tqdm
from torchvision.io import write_png
from torch.utils.data import DataLoader, Subset

from src.agent.recycle_gan import ReCycleGAN
from src.data import Img2ImgDataset
from src.utils.datasets import denormalize, get_ds_from_path
from src.utils.train import read_config

print("##### Reading configuration.")
root = '../../../results/RecycleGAN_CATARACTS_Cataract101_192pix/2022_10_13-06_59_45/'
conf, _ = read_config(root + 'config.yml')
conf.device = 'cuda:1'
conf.data.seq_frames_test = 10
conf.testing.batch_size = 4
conf.data.num_workers = 1
conf.data.dt = 20

N_SAMPLES = 125

print("##### Loading data.")
train_ds_A, test_ds_A = get_ds_from_path(path=conf.data.root_A, conf=conf)
train_ds_B, test_ds_B = get_ds_from_path(path=conf.data.root_B, conf=conf)
print(f"Domain A --- Training: {len(train_ds_A)} --- Testing: {len(test_ds_A)}")
print(f"Domain B --- Training: {len(train_ds_B)} --- Testing: {len(test_ds_B)}")
test_ds = Img2ImgDataset(ds_A=test_ds_A, ds_B=test_ds_B)
np.random.seed(31245)
random.seed(31245)
torch.manual_seed(31245)
idx = np.random.choice(a=np.arange(0, len(test_ds)), size=N_SAMPLES*conf.testing.batch_size, replace=False)
test_ds = Subset(test_ds, idx)
test_dl = DataLoader(test_ds, batch_size=conf.testing.batch_size, num_workers=conf.data.num_workers,
                     shuffle=False, drop_last=True, pin_memory=True)

print("##### Loading model.")
agent = ReCycleGAN(conf)
agent.netG_A2B.load_state_dict(torch.load(root + "checkpoints/Gen_A2B_ep190.pth", map_location='cpu'))
agent.netG_B2A.load_state_dict(torch.load(root + "checkpoints/Gen_B2A_ep190.pth", map_location='cpu'))

print("##### Generating test samples")
if os.path.exists(root + "test_samples/"):
    shutil.rmtree(root + "test_samples/")
time.sleep(0.1)

test_dl = iter(test_dl)

with torch.no_grad():
    for id in tqdm(range(N_SAMPLES)):

        if id == N_SAMPLES - 1:
            break

        sample = next(test_dl)
        N, T, C, H, W = sample['A'].shape

        for t in range(T):
            img_A = sample['A'][:, t].to(conf.device)
            img_B = sample['B'][:, t].to(conf.device)

            img_BA = agent.netG_B2A(img_B)
            img_AB = agent.netG_A2B(img_A)

            img_A = (denormalize(img_A) * 255.0).to(torch.uint8).to('cpu')
            img_B = (denormalize(img_B) * 255.0).to(torch.uint8).to('cpu')
            img_AB = (denormalize(img_AB) * 255.0).to(torch.uint8).to('cpu')
            img_BA = (denormalize(img_BA) * 255.0).to(torch.uint8).to('cpu')

            for n in range(N):
                os.makedirs(root + f"test_samples/real_A/{sample['id_A'][n]}/", exist_ok=True)
                os.makedirs(root + f"test_samples/real_B/{sample['id_B'][n]}/", exist_ok=True)
                os.makedirs(root + f"test_samples/fake_AB/{sample['id_A'][n]}/", exist_ok=True)
                os.makedirs(root + f"test_samples/fake_BA/{sample['id_B'][n]}/", exist_ok=True)
                write_png(img_A[n],
                          filename=root + f"test_samples/real_A/{sample['id_A'][n]}/{sample['frame_nrs_A'][t][n]}.png")
                write_png(img_B[n],
                          filename=root + f"test_samples/real_B/{sample['id_B'][n]}/{sample['frame_nrs_B'][t][n]}.png")
                write_png(img_BA[n],
                          filename=root + f"test_samples/fake_BA/{sample['id_B'][n]}/{sample['frame_nrs_B'][t][n]}.png")
                write_png(img_AB[n],
                          filename=root + f"test_samples/fake_AB/{sample['id_A'][n]}/{sample['frame_nrs_A'][t][n]}.png")

