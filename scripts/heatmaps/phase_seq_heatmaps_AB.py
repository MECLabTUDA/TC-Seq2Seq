import os

import torch
import torch.nn.functional as nnf
import torchvision.transforms as Tf
import matplotlib.pyplot as plt
import numpy as np

from src.data import CATARACTS
from src.agent.unit import UNIT
from src.agent.recycle_gan import ReCycleGAN
from src.model.phase_classifier_model import PhaseClassifier
from src.utils.train import read_config

DEV = 'cuda:4'
MT_UNIT_ROOT = '../../results/MotionUNIT_CATARACTS_Cataract101_192pix/2022_11_03-12_18_46/'
RECYCLEGAN_ROOT = '../../results/RecycleGAN_CATARACTS_Cataract101_192pix/2022_10_13-06_59_45/'


def discrete_cmap(n: int, base_cmap=None):
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, n))
    cmap_name = base.name + str(n)
    return base.from_list(cmap_name, color_list, n)


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

SPLIT = 'Training'
ds_A = CATARACTS(
    root='/local/scratch/CATARACTS-videos-processed/',
    transforms=Tf.Compose([
        Tf.Resize((192, 192)),
        Tf.Normalize(0.5, 0.5)
    ]),
    dt=10,
    n_seq_frames=1,
    sample_phase_annotations=True,
    phases=[0, 3, 4, 5, 6, 7, 8, 10, 13, 14],
    map_phase_annotations=map_phases_,
    split=SPLIT
)

mt_unit_conf, _ = read_config(MT_UNIT_ROOT + 'config.yml')
mt_unit_conf.device = DEV

recycle_gan_conf, _ = read_config(RECYCLEGAN_ROOT + 'config.yml')
recycle_gan_conf.device = DEV

mt_unit = UNIT(mt_unit_conf)
mt_unit.gen_A.load_state_dict(torch.load(MT_UNIT_ROOT + "checkpoints/gen_A_epoch139.pth", map_location='cpu'))
mt_unit.gen_A.eval()
mt_unit.gen_B.load_state_dict(torch.load(MT_UNIT_ROOT + "checkpoints/gen_B_epoch139.pth", map_location='cpu'))
mt_unit.gen_B.eval()

recycle_gan = ReCycleGAN(recycle_gan_conf)
recycle_gan.netG_A2B.load_state_dict(torch.load(RECYCLEGAN_ROOT + "checkpoints/Gen_A2B_ep190.pth", map_location='cpu'))
recycle_gan.netG_A2B.eval()

m_phase = PhaseClassifier(
    n_seq_frames=3,
    n_classes=11,
).to(DEV)
m_phase.load_state_dict(torch.load('phase_model.pth'))
m_phase.eval()

case_ID = None
phase_gt_seq = None
mt_unit_translated_seq = None
mt_unit_phase_pred_seq = None
recycle_gan_translated_seq = None
recycle_gan_phase_pred_seq = None
for i, sample in enumerate(ds_A):
    frame_nr = sample['frame_nrs'][-1]
    if case_ID is None:
        case_ID = sample['case_id']
    if sample['case_id'] != case_ID or i == len(ds_A)-1:

        sample_target_path = f"CATARACTS/{SPLIT}/{case_ID}/"
        os.makedirs(sample_target_path, exist_ok=True)

        phase_gt_seq = phase_gt_seq.cpu().numpy()
        mt_unit_phase_pred_seq = mt_unit_phase_pred_seq.cpu().numpy()
        recycle_gan_phase_pred_seq = recycle_gan_phase_pred_seq.cpu().numpy()

        plt.figure()
        plt.yticks(np.arange(0, 11))
        plt.scatter(np.arange(0, len(phase_gt_seq)), phase_gt_seq, color='blue')
        plt.scatter(np.arange(0, len(mt_unit_phase_pred_seq)), mt_unit_phase_pred_seq - 0.15, color='orange')
        plt.scatter(np.arange(0, len(recycle_gan_phase_pred_seq)), recycle_gan_phase_pred_seq - 0.3, color='green')
        plt.savefig(sample_target_path + "phases_scatter_AB.svg")
        plt.close()

        fig, ax = plt.subplots(3, 1)
        ax[0].imshow(phase_gt_seq[np.newaxis, :], cmap=discrete_cmap(11, 'jet'), aspect='auto')
        ax[1].imshow(mt_unit_phase_pred_seq[np.newaxis, :], cmap=discrete_cmap(11, 'jet'), aspect='auto')
        ax[2].imshow(recycle_gan_phase_pred_seq[np.newaxis, :], cmap=discrete_cmap(11, 'jet'), aspect='auto')
        plt.savefig(sample_target_path + "phases_heatmap_AB.svg")
        plt.close()

        case_ID = sample['case_id']
        phase_gt_seq = None
        mt_unit_translated_seq = None
        mt_unit_phase_pred_seq = None
        recycle_gan_translated_seq = None
        recycle_gan_phase_pred_seq = None

    img = sample['img_seq'].to(DEV)

    phase = sample['phase_seq']
    phase_gt_seq = phase if phase_gt_seq is None\
        else torch.cat([phase_gt_seq, phase])

    with torch.no_grad():
        h, n = mt_unit.gen_A.encode(img)
        pred = mt_unit.gen_B.decode(h + n)
        mt_unit_translated_seq = pred if mt_unit_translated_seq is None else torch.cat([mt_unit_translated_seq, pred], dim=0)

        pred = recycle_gan.netG_A2B(img)
        recycle_gan_translated_seq = pred if recycle_gan_translated_seq is None else torch.cat(
            [recycle_gan_translated_seq, pred], dim=0)

        if mt_unit_translated_seq.shape[0] >= 3:
            phase_pred = m_phase(nnf.interpolate(mt_unit_translated_seq[-3:], size=(128, 128)).view(1, 9, 128, 128))
            phase_pred = phase_pred.argmax(-1).view(1,)
        else:
            phase_pred = torch.tensor([-1]).to(DEV)
        mt_unit_phase_pred_seq = phase_pred if mt_unit_phase_pred_seq is None\
            else torch.cat([mt_unit_phase_pred_seq, phase_pred])

        if recycle_gan_translated_seq.shape[0] >= 3:
            phase_pred = m_phase(nnf.interpolate(recycle_gan_translated_seq[-3:], size=(128, 128)).view(1, 9, 128, 128))
            phase_pred = phase_pred.argmax(-1).view(1,)
        else:
            phase_pred = torch.tensor([-1]).to(DEV)
        recycle_gan_phase_pred_seq = phase_pred if recycle_gan_phase_pred_seq is None\
            else torch.cat([recycle_gan_phase_pred_seq, phase_pred])

    print(f"\rCase {case_ID} Sample {i} Frame {frame_nr}", end="")
