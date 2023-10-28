import argparse
from src.metrics.FID.fid_score import calculate_fid_given_paths

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='Data root path')
parser.add_argument('--id', type=str, help='id of eval. run', default="")
parser.add_argument('--dev', type=str, help='device', default='cpu')
ops = parser.parse_args()

print(ops.id)

score = calculate_fid_given_paths(
    [ops.path + 'test_samples/real_A/',
     ops.path + 'test_samples/fake_BA'],
    batch_size=32,
    device=ops.dev,
    dims=2048
)

print("FID(BA, A): ", score)

score = calculate_fid_given_paths(
    [ops.path + 'test_samples/real_B/', ops.path + 'test_samples/fake_AB'],
    batch_size=32,
    device=ops.dev,
    dims=2048
)

print("FID(AB, B): ", score)
print()
