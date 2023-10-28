import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from src.data import CATARACTS

cats_train_ds = CATARACTS(
    root='/local/scratch/CATARACTS-videos-processed/',
    split='Training',
    sample_imgs=False,
    sample_phase_annotations=True
)
print("CATARACTS training samples: ", len(cats_train_ds))

vals = np.array([])
for pd_frame in cats_train_ds.phase_annotations.values():
    vals = np.concatenate([vals, pd_frame['Steps'].values])

count, _ = np.histogram(vals, bins=np.arange(0, 20))
print("CATARACTS label count: ", count)

sample_data = pd.DataFrame({
    'phase': ['Idle', 'Toric Marking', 'Implant Ejection', 'Incision', 'Viscodilatation', 'Rhexis', 'Hydrodissection',
              'Nuclear Breaking', 'Phacoemulsification', 'Vitrectomy', 'Irrigation + Aspiration', 'Preparing Implant',
              'Manual Aspiration', 'Implanting', 'Positioning', 'OVD Aspiration', 'Suturing', 'Sealing Control',
              'Wound Hydration'],
    'n_frames': count
})
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 40}

matplotlib.rc('font', **font)
plt.figure(figsize=(15, 15))
sn.barplot(data=sample_data, x='n_frames', y='phase', palette=sn.color_palette("dark:#f2b98cff", n_colors=19),
           orient='h', order=sample_data.sort_values('n_frames').phase)
plt.ylabel('Phase')
plt.xlabel('# Frames')
plt.xticks([50000, 100000])
plt.savefig('CATARACTS.svg')
plt.show()
