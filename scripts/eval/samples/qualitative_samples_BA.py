import os
import glob

import matplotlib
import matplotlib.pyplot as plt
from natsort import natsorted
from PIL import Image

os.makedirs('qual_samples/', exist_ok=True)

root_real_A = '../../../results/CycleGAN_CATARACTS_Cataract101_192pix/2022_10_13-06_59_50/test_samples/real_B/'
root_CycleGAN_BA = '../../../results/CycleGAN_CATARACTS_Cataract101_192pix/2022_10_13-06_59_50/test_samples/fake_BA/'
root_RecycleGAN_BA = '../../../results/RecycleGAN_CATARACTS_Cataract101_192pix/2022_10_13-06_59_45/test_samples/fake_BA/'
root_UNIT_BA = '../../../results/UNIT_CATARACTS_Cataract101_192pix/2022_10_13-08_46_20/test_samples/fake_BA/'
root_OF_UNIT_BA = '../../../results/OF_UNIT_CATARACTS_Cataract101_192pix/2022_10_13-08_46_17/test_samples/fake_BA/'
root_MT_UNIT_BA = '../../../results/MotionUNIT_CATARACTS_Cataract101_192pix/2022_10_13-08_46_18/test_samples/fake_BA/'

real_B_samples = natsorted(glob.glob(root_real_A + '*/*.png'))
CycleGAN_BA_samples = natsorted(glob.glob(root_CycleGAN_BA + '*/*.png'))
RecycleGAN_BA_samples = natsorted(glob.glob(root_RecycleGAN_BA + '*/*.png'))
UNIT_BA_samples = natsorted(glob.glob(root_UNIT_BA + '*/*.png'))
MT_UNIT_BA_samples = natsorted(glob.glob(root_MT_UNIT_BA + '*/*.png'))
OF_UNIT_BA_samples = natsorted(glob.glob(root_OF_UNIT_BA + '*/*.png'))

print(len(real_B_samples))

assert len(real_B_samples) == len(CycleGAN_BA_samples) == len(RecycleGAN_BA_samples) == len(UNIT_BA_samples) \
    == len(MT_UNIT_BA_samples) == len(OF_UNIT_BA_samples), \
    f"{len(real_B_samples)}!={len(CycleGAN_BA_samples)}!={len(RecycleGAN_BA_samples)}!={len(UNIT_BA_samples)}!={len(MT_UNIT_BA_samples)}!={len(OF_UNIT_BA_samples)}"

start = 0
N = 20
T = 10
dt = 2
dN = 200

for i in range(start, start+dN*N, dN):


    fig, ax = plt.subplots(6, T, figsize=(T*3, 6*3))
    font = {'weight': 'bold',
            'size': 22}
    matplotlib.rc('font', **font)
    ax[0, 0].set_ylabel('Real Sample')
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[1, 0].set_ylabel('CycleGAN')
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[2, 0].set_ylabel('RecycleGAN')
    ax[2, 0].set_xticks([])
    ax[2, 0].set_yticks([])
    ax[3, 0].set_ylabel('UNIT')
    ax[3, 0].set_xticks([])
    ax[3, 0].set_yticks([])
    ax[4, 0].set_ylabel('OF-UNIT')
    ax[4, 0].set_xticks([])
    ax[4, 0].set_yticks([])
    ax[5, 0].set_ylabel('MT-UNIT')
    ax[5, 0].set_xticks([])
    ax[5, 0].set_yticks([])
    for t in range(T):
        ax[0, t].imshow(Image.open(real_B_samples[i + t * dt]))
        ax[1, t].imshow(Image.open(CycleGAN_BA_samples[i + t * dt]))
        ax[2, t].imshow(Image.open(RecycleGAN_BA_samples[i + t * dt]))
        ax[3, t].imshow(Image.open(UNIT_BA_samples[i + t * dt]))
        ax[4, t].imshow(Image.open(OF_UNIT_BA_samples[i + t * dt]))
        ax[5, t].imshow(Image.open(MT_UNIT_BA_samples[i + t * dt]))
        if t > 0:
            ax[0, t].axis('off')
            ax[1, t].axis('off')
            ax[2, t].axis('off')
            ax[3, t].axis('off')
            ax[4, t].axis('off')
            ax[5, t].axis('off')

    plt.autoscale()
    plt.tight_layout()
    plt.savefig(f'qual_samples/samples_BA_n{i}.svg')
    plt.close()