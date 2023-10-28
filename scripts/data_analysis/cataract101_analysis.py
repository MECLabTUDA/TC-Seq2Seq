import torch
from torch.utils.data import ConcatDataset
import tqdm
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from src.data import Cataract101

cat101_train_ds = Cataract101(
    root='/local/scratch/cataract-101-processed/',
    split='Training',
    sample_imgs=False,
    sample_phase_annotations=True
)
print("Cataract101 training samples: ", len(cat101_train_ds))

cat101_val_ds = Cataract101(
    root='/local/scratch/cataract-101-processed/',
    split='Validation',
    sample_imgs=False,
    sample_phase_annotations=True
)
print("Cataract101 validation samples: ", len(cat101_val_ds))

cat101_test_ds = Cataract101(
    root='/local/scratch/cataract-101-processed/',
    split='Test',
    sample_imgs=False,
    sample_phase_annotations=True
)
print("Cataract101 test samples: ", len(cat101_test_ds))

count = np.zeros(shape=(11,))
for sample in tqdm.tqdm(cat101_train_ds):
    phase = sample['phase_seq']
    count[torch.argmax(phase, dim=-1).item()] += 1
print("Label count: ", count)

sample_data = pd.DataFrame({
    'phase': ['Idle', 'Incision', 'Viscous Agent Injection', 'Rhexis', 'Hydrodissection',
              'Phacoemulsification', 'Irrigation + Aspiration', 'Capsule Polishing', 'Lens Implant Setting-Up',
              'Viscous Agent Removal', 'Tonifying + Antibiotics'],
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
plt.savefig('Cataract101.svg')
plt.show()
