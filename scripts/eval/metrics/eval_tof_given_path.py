import os

import argparse
import numpy as np
from PIL import Image
import torch
from natsort import natsorted
import torchvision.transforms as Tf
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

print(ops.id)
TT = Tf.ToTensor()

#
#   A -> B
#

error_per_seq = []
for seq_A_path, fake_seq_AB_path in tqdm(zip(real_A_seqs, fake_AB_seqs), total=len(real_A_seqs)):

    seq_A_frame_paths = [seq_A_path + frame for frame in natsorted(os.listdir(seq_A_path))]
    fake_AB_frame_paths = [fake_seq_AB_path + frame for frame in natsorted(os.listdir(fake_seq_AB_path))]

    sum_of_errors = 0
    for t in range(1, len(seq_A_frame_paths)):

        # Read consecutive frames
        prv = TT(Image.open(seq_A_frame_paths[t - 1])).unsqueeze(0).to(ops.dev)
        nxt = TT(Image.open(seq_A_frame_paths[t])).unsqueeze(0).to(ops.dev)
        fake_prv = TT(Image.open(fake_AB_frame_paths[t - 1])).unsqueeze(0).to(ops.dev)
        fake_nxt = TT(Image.open(fake_AB_frame_paths[t])).unsqueeze(0).to(ops.dev)
        N, C, H, W = prv.shape

        # Compute flow
        prv_batch, nxt_batch = transforms(prv, nxt)
        fake_prv_batch, fake_nxt_batch = transforms(fake_prv, fake_nxt)
        with torch.no_grad():
            list_of_flows_real = model(prv_batch, nxt_batch)
            list_of_flows_fake = model(fake_prv_batch, fake_nxt_batch)
        predicted_flows_real = list_of_flows_real[-1]
        predicted_flows_fake = list_of_flows_fake[-1]

        p = 1
        sum_of_errors += torch.norm(predicted_flows_fake - predicted_flows_real, p=p)/(H*W*2)

    error_per_seq.append(sum_of_errors.item() / len(seq_A_frame_paths))

print("A -> B: ", np.mean(error_per_seq))

#
#   B -> A
#

error_per_seq = []
for seq_B_path, fake_seq_BA_path in tqdm(zip(real_B_seqs, fake_BA_seqs), total=len(real_B_seqs)):

    seq_B_frame_paths = [seq_B_path + frame for frame in natsorted(os.listdir(seq_B_path))]
    fake_BA_frame_paths = [fake_seq_BA_path + frame for frame in natsorted(os.listdir(fake_seq_BA_path))]

    sum_of_errors = 0
    for t in range(1, len(seq_B_frame_paths)):

        # Read consecutive frames
        prv = TT(Image.open(seq_B_frame_paths[t-1])).unsqueeze(0).to(ops.dev)
        nxt = TT(Image.open(seq_B_frame_paths[t])).unsqueeze(0).to(ops.dev)
        fake_prv = TT(Image.open(fake_BA_frame_paths[t - 1])).unsqueeze(0).to(ops.dev)
        fake_nxt = TT(Image.open(fake_BA_frame_paths[t])).unsqueeze(0).to(ops.dev)

        # Compute flow
        prv_batch, nxt_batch = transforms(prv, nxt)
        fake_prv_batch, fake_nxt_batch = transforms(fake_prv, fake_nxt)
        with torch.no_grad():
            list_of_flows_real = model(prv_batch, nxt_batch)
            list_of_flows_fake = model(fake_prv_batch, fake_nxt_batch)
        predicted_flows_real = list_of_flows_real[-1]
        predicted_flows_fake = list_of_flows_fake[-1]

        sum_of_errors += torch.norm(predicted_flows_fake - predicted_flows_real, p=p)/(H*W*2)

    error_per_seq.append(sum_of_errors.item()/len(seq_B_frame_paths))

print("B -> A: ", np.mean(error_per_seq))

