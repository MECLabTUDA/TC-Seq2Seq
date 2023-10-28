import os
import time

import argparse
import numpy as np
from PIL import Image
import torch
from natsort import natsorted
import torchvision.transforms as Tf
from tqdm import tqdm

from src.metrics.temporal_consistency import temporal_consistency_metric

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='Data root path')
parser.add_argument('--id', type=str, help='id of eval. run', default="")
parser.add_argument('--dev', type=str, help='device', default='cpu')
ops = parser.parse_args()

print(ops.id)
real_A_seqs = [ops.path + "test_samples/real_A/" + case + "/" for case in natsorted(os.listdir(ops.path + "test_samples/real_A/"))]
real_B_seqs = [ops.path + "test_samples/real_B/" + case + "/" for case in natsorted(os.listdir(ops.path + "test_samples/real_B/"))]
fake_BA_seqs = [ops.path + "test_samples/fake_BA/" + case + "/" for case in natsorted(os.listdir(ops.path + "test_samples/fake_BA/"))]
fake_AB_seqs = [ops.path + "test_samples/fake_AB/" + case + "/" for case in natsorted(os.listdir(ops.path + "test_samples/fake_AB/"))]
assert len(real_A_seqs) == len(fake_AB_seqs)
assert len(real_B_seqs) == len(fake_BA_seqs)

TT = Tf.ToTensor()

#
#   A -> B
#

ssim_residual_error_per_seq = []
rmse_residual_error_per_seq = []
for seq_A_path, fake_seq_AB_path in tqdm(zip(real_A_seqs, fake_AB_seqs), total=len(real_A_seqs)):

    seq_A_frame_paths = [seq_A_path + frame for frame in natsorted(os.listdir(seq_A_path))]
    fake_AB_frame_paths = [fake_seq_AB_path + frame for frame in natsorted(os.listdir(fake_seq_AB_path))]

    seq_A = torch.concat([TT(Image.open(path)).unsqueeze(0).to(ops.dev) for path in seq_A_frame_paths]).unsqueeze(0)
    fake_AB = torch.concat([TT(Image.open(path)).unsqueeze(0).to(ops.dev) for path in fake_AB_frame_paths]).unsqueeze(0)

    ssim_residual_error_per_seq.append(temporal_consistency_metric(seq_A, fake_AB, 10, 'SSIM')[0])
    rmse_residual_error_per_seq.append(temporal_consistency_metric(seq_A, fake_AB, 10, 'RMSE')[0])

time.sleep(0.2)
print("A -> B SSIM: ", np.mean(ssim_residual_error_per_seq))
print("A -> B RMSE: ", np.mean(rmse_residual_error_per_seq))
time.sleep(0.2)

#
#   B -> A
#

ssim_residual_error_per_seq = []
rmse_residual_error_per_seq = []
for seq_B_path, fake_seq_BA_path in tqdm(zip(real_B_seqs, fake_BA_seqs), total=len(real_B_seqs)):

    seq_B_frame_paths = [seq_B_path + frame for frame in natsorted(os.listdir(seq_B_path))]
    fake_BA_frame_paths = [fake_seq_BA_path + frame for frame in natsorted(os.listdir(fake_seq_BA_path))]

    seq_B = torch.concat([TT(Image.open(path)).unsqueeze(0).to(ops.dev) for path in seq_B_frame_paths]).unsqueeze(0)
    fake_BA = torch.concat([TT(Image.open(path)).unsqueeze(0).to(ops.dev) for path in fake_BA_frame_paths]).unsqueeze(0)

    ssim_residual_error_per_seq.append(temporal_consistency_metric(seq_B, fake_BA, 10, 'SSIM')[0])
    rmse_residual_error_per_seq.append(temporal_consistency_metric(seq_B, fake_BA, 10, 'RMSE')[0])

time.sleep(0.2)
print("B -> A SSIM: ", np.mean(ssim_residual_error_per_seq))
print("B -> A RMSE: ", np.mean(rmse_residual_error_per_seq))
time.sleep(0.2)
