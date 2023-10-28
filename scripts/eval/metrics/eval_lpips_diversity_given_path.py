import glob
import lpips
import argparse
import numpy as np
from torchvision.io import read_image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='Data root path')
parser.add_argument('--id', type=str, help='id of eval. run', default="")
parser.add_argument('--dev', type=str, help='device', default='cpu')
ops = parser.parse_args()

loss_fn_alex = lpips.LPIPS(net='alex').to(ops.dev)

print(ops.id)

#
# BA
#

fake_seq_paths = glob.glob(ops.path + "test_samples/fake_BA/*/*.png")
shuffled = fake_seq_paths.copy()
np.random.shuffle(shuffled)
dists = []
for paths in tqdm(zip(fake_seq_paths, shuffled), total=len(shuffled)):
    dists.append(loss_fn_alex(read_image(paths[0]).to(ops.dev), read_image(paths[1]).to(ops.dev)).item())
print("LPIPS(BA): ", (np.mean(dists), np.var(dists)))

#
# AB
#

fake_seq_paths = glob.glob(ops.path + "test_samples/fake_AB/*/*.png")
shuffled = fake_seq_paths.copy()
np.random.shuffle(shuffled)
dists = []
for paths in tqdm(zip(fake_seq_paths, shuffled), total=len(shuffled)):
    dists.append(loss_fn_alex(read_image(paths[0]).to(ops.dev), read_image(paths[1]).to(ops.dev)).item())
print("LPIPS(AB): ", (np.mean(dists), np.var(dists)))
