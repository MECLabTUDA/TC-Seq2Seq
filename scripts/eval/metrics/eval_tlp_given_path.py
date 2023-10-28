import os

import argparse
import numpy as np
from PIL import Image
import lpips
from natsort import natsorted
import torchvision.transforms as Tf
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='Data root path')
parser.add_argument('--id', type=str, help='id of eval. run', default="")
parser.add_argument('--dev', type=str, help='device', default='cpu')
ops = parser.parse_args()

loss_fn_alex = lpips.LPIPS(net='alex').to(ops.dev)

real_A_seqs = [ops.path + "test_samples/real_A/" + case + "/" for case in natsorted(os.listdir(ops.path + "test_samples/real_A/"))]
real_B_seqs = [ops.path + "test_samples/real_B/" + case + "/" for case in natsorted(os.listdir(ops.path + "test_samples/real_B/"))]
fake_BA_seqs = [ops.path + "test_samples/fake_BA/" + case + "/" for case in natsorted(os.listdir(ops.path + "test_samples/fake_BA/"))]
fake_AB_seqs = [ops.path + "test_samples/fake_AB/" + case + "/" for case in natsorted(os.listdir(ops.path + "test_samples/fake_AB/"))]
assert len(real_A_seqs) == len(fake_AB_seqs)
assert len(real_B_seqs) == len(fake_BA_seqs)

TT = Tf.ToTensor()

print(ops.id)

#
#   A -> B
#

mean_error_per_seq = []
for seq_A_path, fake_seq_AB_path in tqdm(zip(real_A_seqs, fake_AB_seqs), total=len(real_A_seqs)):

    seq_A_frame_paths = [seq_A_path + frame for frame in natsorted(os.listdir(seq_A_path))]
    fake_AB_frame_paths = [fake_seq_AB_path + frame for frame in natsorted(os.listdir(fake_seq_AB_path))]

    error_per_frame = []
    for t in range(1, len(seq_A_frame_paths)):

        # Read consecutive frames
        prv = TT(Image.open(seq_A_frame_paths[t - 1])).unsqueeze(0).to(ops.dev)
        nxt = TT(Image.open(seq_A_frame_paths[t])).unsqueeze(0).to(ops.dev)
        fake_prv = TT(Image.open(fake_AB_frame_paths[t - 1])).unsqueeze(0).to(ops.dev)
        fake_nxt = TT(Image.open(fake_AB_frame_paths[t])).unsqueeze(0).to(ops.dev)
        N, C, H, W = prv.shape

        d_real = loss_fn_alex(prv, nxt)
        d_fake = loss_fn_alex(fake_prv, fake_nxt)

        error_per_frame.append(np.abs(d_real.item() - d_fake.item()))

    mean_error_per_seq.append(np.mean(error_per_frame))

print("A -> B: ", np.mean(mean_error_per_seq))

#
#   B -> A
#

error_per_seq = []
for seq_B_path, fake_seq_BA_path in tqdm(zip(real_B_seqs, fake_BA_seqs), total=len(real_B_seqs)):

    seq_B_frame_paths = [seq_B_path + frame for frame in natsorted(os.listdir(seq_B_path))]
    fake_BA_frame_paths = [fake_seq_BA_path + frame for frame in natsorted(os.listdir(fake_seq_BA_path))]

    error_per_frame = []
    for t in range(1, len(seq_B_frame_paths)):
        # Read consecutive frames
        prv = TT(Image.open(seq_B_frame_paths[t - 1])).unsqueeze(0).to(ops.dev)
        nxt = TT(Image.open(seq_B_frame_paths[t])).unsqueeze(0).to(ops.dev)
        fake_prv = TT(Image.open(fake_BA_frame_paths[t - 1])).unsqueeze(0).to(ops.dev)
        fake_nxt = TT(Image.open(fake_BA_frame_paths[t])).unsqueeze(0).to(ops.dev)
        N, C, H, W = prv.shape

        d_real = loss_fn_alex(prv, nxt)
        d_fake = loss_fn_alex(fake_prv, fake_nxt)

        error_per_frame.append(np.abs(d_real.item() - d_fake.item()))

    mean_error_per_seq.append(np.mean(error_per_frame))

print("B -> A: ", np.mean(mean_error_per_seq))

