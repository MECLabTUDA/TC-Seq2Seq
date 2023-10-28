import torch
import tqdm
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, Subset

from src.data import CATARACTS, Cataract101

def _map_phases_(t: torch.Tensor):
    pass

cat101_train_ds = Cataract101(
    root='/local/scratch/cataract-101-processed/',
    split='Training',
    sample_imgs=False,
    sample_phase_annotations=True,
    map_phase_annotations=_map_phases_
)
print("Cataract101 training samples: ", len(cat101_train_ds))

cat101_val_ds = Cataract101(
    root='/local/scratch/cataract-101-processed/',
    split='Validation',
    sample_imgs=False,
    sample_phase_annotations=True,
    map_phase_annotations = _map_phases_
)
print("Cataract101 validation samples: ", len(cat101_val_ds))
cat101_test_ds = Cataract101(
    root='/local/scratch/cataract-101-processed/',
    split='Test',
    sample_imgs=False,
    sample_phase_annotations=True,
    map_phase_annotations = _map_phases_
)
print("Cataract101 test samples: ", len(cat101_test_ds))


def _map_phases_(t: torch.Tensor):
    t[t == 3] = 1
    t[t == 4] = 2
    t[t == 5] = 3
    t[t == 6] = 4
    t[t == 7] = 5
    t[t == 8] = 5
    t[t == 10] = 6
    t[t == 13] = 8
    t[t == 14] = 8


cats_train_to_cat101_ds = CATARACTS(
    root='/local/scratch/CATARACTS-to-Cataract101/',
    phases=[3, 4, 5, 6, 10, 13, 14],
    split='Training',
    sample_imgs=False,
    sample_phase_annotations=True,
    map_phase_annotations=_map_phases_,
)
print("Gen train samples: ", len(cats_train_to_cat101_ds))

cats_val_to_cat101_ds = CATARACTS(
    root='/local/scratch/CATARACTS-to-Cataract101/',
    phases=[3, 4, 5, 6, 10, 13, 14],
    split='Validation',
    sample_imgs=False,
    sample_phase_annotations=True,
    map_phase_annotations=_map_phases_,
)
print("Gen val samples: ", len(cats_val_to_cat101_ds))
cats_test_to_cat101_ds = CATARACTS(
    root='/local/scratch/CATARACTS-to-Cataract101/',
    phases=[3, 4, 5, 6, 10, 13, 14],
    split='Test',
    sample_imgs=False,
    sample_phase_annotations=True,
    map_phase_annotations=_map_phases_,
)
print("Gen test samples: ", len(cats_test_to_cat101_ds))

train_ds = ConcatDataset([cat101_train_ds, cats_train_to_cat101_ds, cats_test_to_cat101_ds])

count = np.zeros(shape=(11,))
for sample in tqdm.tqdm(train_ds):
    phase = sample['phase_seq']
    count[phase.item()] += 1

print("Label count: ", count)

sample_data = pd.DataFrame({
    'phase': ['Idle', 'Incision', 'Viscous Agent Injection', 'Rhexis', 'Hydrodissection', 'Phacoemulsification',
              'Irrigation + Aspiration', 'Capsule Polishing', 'Lens Implant Setting-Up', 'Viscous Agent Removal',
              'Tonifying + Antibiotics'],
    'n_frames': count
})
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 40}

matplotlib.rc('font', **font)
plt.figure(figsize=(15, 15))
sn.barplot(data=sample_data, x='n_frames', y='phase', palette=sn.color_palette("dark:#8cf2d3ed", n_colors=11), orient='h', order=sample_data.sort_values('n_frames').phase)
plt.ylabel('Phase')
plt.xlabel('# Frames')
plt.xticks([0, 50000, 100000, 150000, 200000])
plt.autoscale()
plt.savefig('ExtendedCataract101.svg')
plt.show()
