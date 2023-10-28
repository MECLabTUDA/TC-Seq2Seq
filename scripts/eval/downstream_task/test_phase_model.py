import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as Tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data import Cataract101
from src.model.phase_classifier_model import PhaseClassifier

dev = 'cuda'

train_ds = Cataract101(root='/local/scratch/cataract-101-processed/',
                       n_seq_frames=3,
                       dt=1,
                       transforms=Tf.Compose([
                           Tf.Resize((128, 128)),
                           Tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]),
                       sample_phase_annotations=True,
                       split="Training")

val_ds = Cataract101(root='/local/scratch/cataract-101-processed/',
                     n_seq_frames=3,
                     dt=1,
                     transforms=Tf.Compose([
                         Tf.Resize((128, 128)),
                         Tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                     ]),
                     sample_phase_annotations=True,
                     split="Validation")

test_ds = Cataract101(root='/local/scratch/cataract-101-processed/',
                      n_seq_frames=3,
                      dt=1,
                      transforms=Tf.Compose([
                          Tf.Resize((128, 128)),
                          Tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ]),
                      sample_phase_annotations=True,
                      split="Test")

test_dl = DataLoader(val_ds, batch_size=1, num_workers=1, shuffle=False)

m = PhaseClassifier(n_seq_frames=3, n_classes=11).to(dev)
m.load_state_dict(torch.load('../../../results/phase_model_decay_lr/phase_model_extended1.pth', map_location='cpu'))

phase_predictions = None
phase_target = None
case_id = None
with torch.no_grad():
    for id, sample in enumerate(tqdm(test_dl)):
        img = sample['img_seq']
        N, T, C, H, W = img.shape
        _case_id = sample['case_id']

        if case_id is None:
            case_id = _case_id
        elif _case_id != case_id:
            break

        img = img.view((N, T * C, H, W)).to(dev)
        phase = sample['phase_seq'][:, -1]
        predicted_phase = m(img)
        phase_predictions = predicted_phase if phase_predictions is None else torch.cat(
            [phase_predictions, predicted_phase], dim=0)
        phase_target = phase if phase_target is None else torch.cat([phase_target, phase], dim=0)

phase_predictions = torch.argmax(phase_predictions, dim=-1)
phase_target = torch.argmax(phase_target, dim=-1)
plt.figure()
plt.grid()
plt.scatter(np.arange(0, len(phase_target)), phase_target.cpu().numpy(), label='target')
plt.scatter(np.arange(0, len(phase_predictions)), phase_predictions.cpu().numpy(), label='prediction')
plt.ylim(0, 11)
plt.legend()
plt.show()
