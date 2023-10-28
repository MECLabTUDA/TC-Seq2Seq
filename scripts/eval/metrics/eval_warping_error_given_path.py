import os
import sys

import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from natsort import natsorted
import torchvision.transforms as Tf
from torchvision.utils import flow_to_image
from torchvision.io import read_image
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='Data root path')
parser.add_argument('--id', type=str, help='id of eval. run', default="")
parser.add_argument('--dev', type=str, help='device', default='cpu')
ops = parser.parse_args()

real_A_seqs = [ops.path + "test_samples/real_A/" + case + "/" for case in natsorted(os.listdir(ops.path + "test_samples/real_A/"))]
real_B_seqs = [ops.path + "test_samples/real_B/" + case + "/" for case in natsorted(os.listdir(ops.path + "test_samples/real_B/"))]
fake_BA_seqs = [ops.path + "test_samples/fake_BA/" + case + "/" for case in natsorted(os.listdir(ops.path + "test_samples/fake_BA/"))]
fake_AB_seqs = [ops.path + "test_samples/fake_AB/" + case + "/" for case in natsorted(os.listdir(ops.path + "test_samples/fake_AB/"))]
assert len(real_A_seqs) == len(fake_AB_seqs)
assert len(real_B_seqs) == len(fake_BA_seqs)

weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()
model = raft_large(weights=weights, progress=False).to(ops.dev)
model = model.eval()

TT = Tf.ToTensor()
print(ops.id)

#
#   A -> B
#

sum_of_errors_per_seq = []
for seq_A_path, fake_seq_AB_path in tqdm(zip(real_A_seqs, fake_AB_seqs), total=len(real_A_seqs)):

    seq_A_frame_paths = [seq_A_path + frame for frame in natsorted(os.listdir(seq_A_path))]
    fake_AB_frame_paths = [fake_seq_AB_path + frame for frame in natsorted(os.listdir(fake_seq_AB_path))]

    warping_error_per_frame = 0
    for t in range(1, len(seq_A_frame_paths)):

        # Read consecutive frames
        prv = TT(Image.open(seq_A_frame_paths[t-1])).unsqueeze(0).to(ops.dev)
        nxt = TT(Image.open(seq_A_frame_paths[t])).unsqueeze(0).to(ops.dev)

        # Compute flow
        prv_batch, nxt_batch = transforms(prv, nxt)
        with torch.no_grad():
            list_of_flows = model(prv_batch, nxt_batch)
        predicted_flows = list_of_flows[-1]
        flow_imgs = flow_to_image(predicted_flows)

        # Warp translated image
        fake_prv = read_image(fake_AB_frame_paths[t - 1]).unsqueeze(0).to(ops.dev)/255.0
        fake_nxt = read_image(fake_AB_frame_paths[t]).unsqueeze(0).to(ops.dev)/255.0
        fake_prv_batch, fake_nxt_batch = transforms(fake_prv, fake_nxt)
        N, C, H, W = fake_nxt.shape

        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(N, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(N, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().to(ops.dev)
        vgrid = Variable(grid) + predicted_flows
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        flo = predicted_flows.permute(0, 2, 3, 1)
        output = F.grid_sample(fake_nxt_batch, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(fake_nxt_batch.size())).to(ops.dev)
        mask = F.grid_sample(mask, vgrid, align_corners=True)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        warped_fake_prv = mask * output

        warping_error_per_frame += torch.sqrt(F.mse_loss(warped_fake_prv, fake_prv_batch))

    sum_of_errors_per_seq.append(warping_error_per_frame.item()/len(seq_A_frame_paths))

print("A -> B: ", np.mean(sum_of_errors_per_seq))

#
#   B -> A
#

sum_of_errors_per_seq = []
for seq_B_path, fake_seq_BA_path in tqdm(zip(real_B_seqs, fake_BA_seqs), total=len(real_B_seqs)):

    seq_B_frame_paths = [seq_B_path + frame for frame in natsorted(os.listdir(seq_B_path))]
    fake_BA_frame_paths = [fake_seq_BA_path + frame for frame in natsorted(os.listdir(fake_seq_BA_path))]

    warping_error_per_frame = 0
    for t in range(1, len(seq_B_frame_paths)):

        # Read consecutive frames
        prv = TT(Image.open(seq_B_frame_paths[t-1])).unsqueeze(0).to(ops.dev)
        nxt = TT(Image.open(seq_B_frame_paths[t])).unsqueeze(0).to(ops.dev)

        # Compute flow
        prv_batch, nxt_batch = transforms(prv, nxt)
        with torch.no_grad():
            list_of_flows = model(prv_batch, nxt_batch)
        predicted_flows = list_of_flows[-1]
        flow_imgs = flow_to_image(predicted_flows)

        # Warp translated image
        fake_prv = read_image(fake_BA_frame_paths[t - 1]).unsqueeze(0).to(ops.dev)/255.0
        fake_nxt = read_image(fake_BA_frame_paths[t]).unsqueeze(0).to(ops.dev)/255.0
        fake_prv_batch, fake_nxt_batch = transforms(fake_prv, fake_nxt)
        N, C, H, W = fake_nxt.shape

        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(N, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(N, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().to(ops.dev)
        vgrid = Variable(grid) + predicted_flows
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        flo = predicted_flows.permute(0, 2, 3, 1)
        output = F.grid_sample(fake_nxt_batch, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(fake_nxt_batch.size())).to(ops.dev)
        mask = F.grid_sample(mask, vgrid, align_corners=True)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        warped_fake_prv = mask * output

        warping_error_per_frame += torch.sqrt(F.mse_loss(warped_fake_prv, fake_prv_batch))

    sum_of_errors_per_seq.append(warping_error_per_frame.item()/len(seq_B_frame_paths))

print("B -> A: ", np.mean(sum_of_errors_per_seq))

