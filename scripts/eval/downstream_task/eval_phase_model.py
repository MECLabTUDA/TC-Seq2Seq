import argparse

import torch
import torchvision.transforms as Tf
import numpy as np
from torchmetrics import F1Score, AUROC, AveragePrecision
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.phase_classifier_model import PhaseClassifier
from src.data import CATARACTS

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str, help='Data root path')
parser.add_argument('--modelpath', type=str, help='Phase classifier model path')
parser.add_argument('--id', type=str, help='id of eval. run', default='')
parser.add_argument('--dev', type=str, help='device', default='cpu')
ops = parser.parse_args()

print("##### Loading data.")


def map_phases_(t: torch.Tensor):
    t[t == 3] = 1
    t[t == 4] = 2
    t[t == 5] = 3
    t[t == 6] = 4
    t[t == 7] = 5
    t[t == 8] = 5
    t[t == 10] = 6
    t[t == 13] = 8
    t[t == 14] = 8


test_ds = CATARACTS(
    root=ops.datapath + 'test_samples/fake_AB/',
    n_seq_frames=3,
    dt=1,
    split='Test',
    sample_phase_annotations=True,
    file_end='.png',
    transforms=Tf.Compose([
        Tf.Resize((128, 128)),
        Tf.Normalize(0.5, 0.5)
    ]),
    phases=[3, 5, 6, 8, 10, 13, 14],
    # phases=[0, 3, 4, 5, 6, 7, 8, 10, 13, 14],
    map_phase_annotations=map_phases_
)
test_dl = DataLoader(test_ds, batch_size=4, num_workers=4, shuffle=False)


print("##### Loading model")
m = PhaseClassifier(n_seq_frames=3, n_classes=11).to(ops.dev)
m.load_state_dict(torch.load(ops.modelpath, map_location='cpu'))
m.eval()

print("##### Evaluating on translated data:")
phase_predictions = None
phase_target = None
with torch.no_grad():
    for id, sample in enumerate(tqdm(test_dl)):
        img = sample['img_seq']
        N, T, C, H, W = img.shape
        img = img.view((N, T * C, H, W)).to(ops.dev)
        phase = sample['phase_seq'][:, -1]
        predicted_phase = m(img)
        phase_predictions = predicted_phase if phase_predictions is None \
            else torch.cat([phase_predictions, predicted_phase], dim=0)
        phase_target = phase if phase_target is None else torch.cat([phase_target, phase], dim=0)

ap_score = AveragePrecision(num_classes=11).to(ops.dev)
f1_score = F1Score(num_classes=11).to(ops.dev)
auroc = AUROC(num_classes=11).to(ops.dev)
ap_s = ap_score(phase_predictions, phase_target.to(ops.dev))
f1_s = f1_score(phase_predictions, phase_target.to(ops.dev))
auroc_s = auroc(phase_predictions, phase_target.to(ops.dev))

ap_score = AveragePrecision(num_classes=11, average=None).to(ops.dev)
f1_score = F1Score(num_classes=11, average=None).to(ops.dev)
auroc = AUROC(num_classes=11, average=None).to(ops.dev)
ap_v = torch.stack(ap_score(phase_predictions, phase_target.to(ops.dev))).cpu().numpy()
ap_v = np.nan_to_num(ap_v)
f1_v = f1_score(phase_predictions, phase_target.to(ops.dev)).cpu().numpy()
auroc_v = auroc(phase_predictions, phase_target.to(ops.dev)).cpu().numpy()

print(f"{ops.id}\t"
      f"AP: {ap_s}\t Class var.: {np.var(ap_v)}\t"
      f"F1: {f1_s}\tClass var.: {np.var(f1_v)}\t"
      f"AUROC: {auroc_s}\tClass var.: {np.var(auroc_v)}")

