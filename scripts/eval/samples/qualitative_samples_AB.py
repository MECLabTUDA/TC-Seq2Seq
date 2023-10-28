import glob
import os

import matplotlib
import matplotlib.pyplot as plt
from natsort import natsorted
from PIL import Image

os.makedirs('qual_samples/', exist_ok=True)

root_real_A = '../../../results/CycleGAN_CATARACTS_Cataract101_192pix/2022_10_13-06_59_50/test_samples/real_A/'
root_CycleGAN_AB = '../../../results/CycleGAN_CATARACTS_Cataract101_192pix/2022_10_13-06_59_50/test_samples/fake_AB/'
root_RecycleGAN_AB = '../../../results/RecycleGAN_CATARACTS_Cataract101_192pix/2022_10_13-06_59_45/test_samples/fake_AB/'
root_UNIT_AB = '../../../results/UNIT_CATARACTS_Cataract101_192pix/2022_10_13-08_46_20/test_samples/fake_AB/'
root_OF_UNIT_AB = '../../../results/OF_UNIT_CATARACTS_Cataract101_192pix/2022_10_13-08_46_17/test_samples/fake_AB/'
root_MT_UNIT_AB = '../../../results/MotionUNIT_CATARACTS_Cataract101_192pix/2022_10_13-08_46_18/test_samples/fake_AB/'

real_A_samples = natsorted(glob.glob(root_real_A + '*/*.png'))
CycleGAN_AB_samples = natsorted(glob.glob(root_CycleGAN_AB + '*/*.png'))
RecycleGAN_AB_samples = natsorted(glob.glob(root_RecycleGAN_AB + '*/*.png'))
UNIT_AB_samples = natsorted(glob.glob(root_UNIT_AB + '*/*.png'))
MT_UNIT_AB_samples = natsorted(glob.glob(root_MT_UNIT_AB + '*/*.png'))
OF_UNIT_AB_samples = natsorted(glob.glob(root_OF_UNIT_AB + '*/*.png'))

print(len(real_A_samples))

assert len(real_A_samples) == len(CycleGAN_AB_samples) == len(RecycleGAN_AB_samples) == len(UNIT_AB_samples) \
    == len(MT_UNIT_AB_samples) == len(OF_UNIT_AB_samples), \
    f"{len(real_A_samples)}!={len(CycleGAN_AB_samples)}!={len(RecycleGAN_AB_samples)}!={len(UNIT_AB_samples)}!={len(MT_UNIT_AB_samples)}!={len(OF_UNIT_AB_samples)}"

start = 0
N = 20
T = 10
dt = 2
dN = 100

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
        ax[0, t].imshow(Image.open(real_A_samples[i + t * dt]))
        ax[1, t].imshow(Image.open(CycleGAN_AB_samples[i + t * dt]))
        ax[2, t].imshow(Image.open(RecycleGAN_AB_samples[i + t * dt]))
        ax[3, t].imshow(Image.open(UNIT_AB_samples[i + t * dt]))
        ax[4, t].imshow(Image.open(OF_UNIT_AB_samples[i + t * dt]))
        ax[5, t].imshow(Image.open(MT_UNIT_AB_samples[i + t * dt]))
        if t > 0:
            ax[0, t].axis('off')
            ax[1, t].axis('off')
            ax[2, t].axis('off')
            ax[3, t].axis('off')
            ax[4, t].axis('off')
            ax[5, t].axis('off')

    plt.autoscale()
    plt.tight_layout()
    plt.savefig(f'qual_samples/samples_AB_n{i}.svg')
    plt.close()
