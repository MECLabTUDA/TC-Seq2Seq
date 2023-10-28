import os
import shutil
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.utils import flow_to_image
from tqdm import tqdm
import numpy as np

from src.agent.unit import UNIT
from src.model.unet import UNet
from src.utils.train import read_config
from src.utils.datasets import get_dataloaders, denormalize

# Read config, make log dirs etc.
conf, uneasy_conf = read_config('src/config/OF_UNIT_CATARACTS_cataract101.yml')
conf.device = 'cuda:2'
conf.data.seq_frames_train = 2
conf.training.batch_size = 4
root = '/gris/gris-f/homestud/yfrisch/swc/Temporal-Consistent-CycleGAN/'
# root = '/home/yannik/Temporal-Consistent-CycleGAN/'
if os.path.isdir(root + 'of_test/'):
    shutil.rmtree(root + 'of_test/')
os.makedirs(root + 'of_test/', exist_ok=False)

train_dl, test_dl = get_dataloaders(conf, shuffle_test=True)

agent = UNIT(conf)
agent.gen_A.load_state_dict(torch.load(
    root + 'results/UNIT_CATARACTS_Cataract101/2022_09_30-21_30_35/checkpoints/gen_A_epoch499.PTH', map_location='cpu'))
agent.gen_B.load_state_dict(torch.load(
    root + 'results/UNIT_CATARACTS_Cataract101/2022_09_30-21_30_35/checkpoints/gen_B_epoch499.PTH', map_location='cpu'))

unet = UNet(n_channels=3+3+3+3+2, n_classes=2).to(conf.device)
optim = torch.optim.AdamW(list(unet.parameters()) +
                          list(agent.gen_A.parameters()) +
                          list(agent.gen_B.parameters()), lr=0.01)
weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()
flo_model = raft_large(weights=weights, progress=False).to(conf.device)
flo_model = flo_model.eval()

for epoch in range(10):

    loss_per_sample = []
    for id, sample in enumerate(tqdm(train_dl)):

        a1 = sample['A'][:, -2].to(conf.device)
        a2 = sample['A'][:, -1].to(conf.device)
        b1 = sample['B'][:, -2].to(conf.device)
        b2 = sample['B'][:, -1].to(conf.device)
        N, C, H, W = b2.shape

        optim.zero_grad()

        # Estimate flow a1 -> a2
        with torch.no_grad():
            a_flow = flo_model(a1, a2)[-1]

            # Translate a1 and a2
            h_a1, n_a1 = agent.gen_A.encode(a1)
            h_a2, n_a2 = agent.gen_A.encode(a2)
            ab1 = agent.gen_B.decode(h_a1 + n_a1)
            ab2 = agent.gen_B.decode(h_a2 + n_a2)

        # Translate motion
        translated_flow = unet(torch.cat([
            a1, a2, ab1, ab2, a_flow
        ], dim=1))

        # Warp translated image according to translated motion
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(N, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(N, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().to(conf.device)
        b_vgrid = Variable(grid, requires_grad=True) - a_flow
        b_vgrid[:, 0, :, :] = 2.0 * b_vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        b_vgrid[:, 1, :, :] = 2.0 * b_vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        b_vgrid = b_vgrid.permute(0, 2, 3, 1)
        ab2_warped = F.grid_sample(ab1, b_vgrid, align_corners=True)
        ab_mask = Variable(torch.ones(ab1.size()), requires_grad=True).to(conf.device)
        ab_mask = F.grid_sample(ab_mask, b_vgrid)
        ab_mask[ab_mask < 0.9999] = 0
        ab_mask[ab_mask > 0] = 1
        ab2_warped = ab_mask * ab2_warped

        loss = F.mse_loss(ab2_warped, ab2)
        loss.backward()
        loss_per_sample.append(loss.item())
        optim.step()

        if id == 10:
            fig, ax = plt.subplots(N, 5)
            for n in range(N):
                ax[n, 0].imshow(denormalize(a1[n]).permute(1, 2, 0).detach().cpu())
                ax[n, 0].axis('off')
                ax[n, 1].imshow(flow_to_image(a_flow[n]).permute(1, 2, 0).detach().cpu())
                ax[n, 1].axis('off')
                ax[n, 2].imshow(denormalize(a2[n]).permute(1, 2, 0).detach().cpu())
                ax[n, 2].axis('off')
                ax[n, 3].imshow(denormalize(ab1[n]).permute(1, 2, 0).detach().cpu())
                ax[n, 3].axis('off')
                ax[n, 4].imshow(denormalize(ab2_warped[n]).permute(1, 2, 0).detach().cpu())
                ax[n, 4].axis('off')
            plt.savefig(root + f'of_test/ep{epoch}.png')
            break

    time.sleep(0.2)
    print(f"Epoch {epoch} Avg. loss {np.mean(loss_per_sample)}")
    time.sleep(0.2)
