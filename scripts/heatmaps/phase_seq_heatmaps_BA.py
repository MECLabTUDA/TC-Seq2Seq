import os

import numpy as np
import torch
import torch.nn.functional as nnf
import torchvision.transforms as Tf
import matplotlib.pyplot as plt

from src.data import Cataract101
from src.agent.unit import UNIT
from src.agent.recycle_gan import ReCycleGAN
from src.model.phase_classifier_model import PhaseClassifier
from src.utils.train import read_config

DEV = 'cuda:5'
MT_UNIT_ROOT = '../../results/MotionUNIT_CATARACTS_Cataract101_192pix/2022_11_03-12_18_46/'
RECYCLEGAN_ROOT = '../../results/RecycleGAN_CATARACTS_Cataract101_192pix/2022_10_13-06_59_45/'


def discrete_cmap(n: int, base_cmap=None):
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, n))
    cmap_name = base.name + str(n)
    return base.from_list(cmap_name, color_list, n)


def map_phases_(t: torch.Tensor):
    pass

SPLIT = 'Validation'
ds_B = Cataract101(
    root='/local/scratch/cataract-101-processed/',
    transforms=Tf.Compose([
        Tf.Resize((192, 192)),
        Tf.Normalize(0.5, 0.5)
    ]),
    dt=10,
    n_seq_frames=1,
    sample_phase_annotations=True,
    map_phase_annotations=map_phases_,
    # split='Training'
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
recycle_gan.netG_A2B.load_state_dict(torch.load(RECYCLEGAN_ROOT + "checkpoints/Gen_B2A_ep190.pth", map_location='cpu'))
recycle_gan.netG_A2B.eval()

m_phase = PhaseClassifier(
    n_seq_frames=3,
    n_classes=11,
).to(DEV)
# m_phase.load_state_dict(torch.load('../../../phase_model.pth'))
m_phase.load_state_dict(torch.load('phase_model.pth'))
m_phase.eval()

m_phase_extended = PhaseClassifier(
    n_seq_frames=3,
    n_classes=11,
).to(DEV)
#m_phase_extended.load_state_dict(torch.load('../../../phase_model_extended1.pth'))
m_phase_extended.load_state_dict(torch.load('phase_model_extended1.pth'))
m_phase_extended.eval()

case_ID = None
img_seq = None
phase_gt_seq = None
phase_pred_seq = None
phase_pred_extended_seq = None
mt_unit_translated_seq = None
recycle_gan_translated_seq = None
for i, sample in enumerate(ds_B):
    frame_nr = sample['frame_nrs'][-1]
    if case_ID is None:
        case_ID = sample['case_id']
    if sample['case_id'] != case_ID or i == len(ds_B)-1:

        sample_target_path = f"Cataract101/{SPLIT}/{case_ID}/"
        os.makedirs(sample_target_path, exist_ok=True)

        phase_gt_seq = phase_gt_seq.cpu().numpy()
        phase_pred_seq = phase_pred_seq.cpu().numpy()
        phase_pred_extended_seq = phase_pred_extended_seq.cpu().numpy()

        plt.figure()
        plt.yticks(np.arange(0, 11))

        plt.scatter(np.arange(0, len(phase_gt_seq)), phase_gt_seq, color='blue')
        plt.scatter(np.arange(0, len(phase_pred_seq)), phase_pred_seq - 0.1, color='orange')
        plt.scatter(np.arange(0, len(phase_pred_extended_seq)), phase_pred_extended_seq - 0.2, color='green')
        plt.savefig(sample_target_path + "phases_scatter_BA.svg")
        plt.close()

        fig, ax = plt.subplots(3, 1)
        ax[0].imshow(phase_gt_seq[np.newaxis, :], cmap=discrete_cmap(11, 'jet'), aspect='auto')
        ax[1].imshow(phase_pred_seq[np.newaxis, :], cmap=discrete_cmap(11, 'jet'), aspect='auto')
        ax[2].imshow(phase_pred_extended_seq[np.newaxis, :], cmap=discrete_cmap(11, 'jet'), aspect='auto')
        plt.savefig(sample_target_path + "phases_heatmap_BA.svg")
        plt.close()

        case_ID = sample['case_id']
        img_seq = None
        phase_gt_seq = None
        phase_pred_seq = None
        phase_pred_extended_seq = None
        mt_unit_translated_seq = None
        recycle_gan_translated_seq = None

    img = sample['img_seq'].to(DEV)
    img_seq = img if img_seq is None else torch.cat([img_seq, img], dim=0)

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

        # Run predictions on source seq
        if img_seq.shape[0] >= 3:
            phase_pred = m_phase(nnf.interpolate(img_seq[-3:], size=(128, 128)).view(1, 9, 128, 128))
            phase_pred = phase_pred.argmax(-1).view(1, )
            phase_pred_extended = m_phase_extended(nnf.interpolate(img_seq[-3:], size=(128, 128)).view(1, 9, 128, 128))
            phase_pred_extended = phase_pred_extended.argmax(-1).view(1, )
        else:
            # phase_pred = torch.tensor([-1]).to(DEV)
            # phase_pred_extended = torch.tensor([-1]).to(DEV)
            phase_pred = torch.tensor([0]).to(DEV)
            phase_pred_extended = torch.tensor([0]).to(DEV)
        phase_pred_seq = phase_pred if phase_pred_seq is None else torch.cat([phase_pred_seq, phase_pred])
        phase_pred_extended_seq = phase_pred_extended if phase_pred_extended_seq is None else torch.cat(
            [phase_pred_extended_seq, phase_pred_extended])

    print(f"\rCase {case_ID} Sample {i} Frame {frame_nr}", end="")
