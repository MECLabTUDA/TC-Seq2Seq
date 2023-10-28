import os

import torch
import torchvision.transforms as Tf
from torchvision.io.video import write_video

from src.data import CATARACTS
from src.agent.unit import UNIT
from src.agent.cycle_gan import CycleGAN_Agent
from src.agent.recycle_gan import ReCycleGAN
from src.utils.train import read_config
from src.utils.datasets import denormalize

DEV = 'cuda:0'
MT_UNIT_ROOT = '../../results/MotionUNIT_CATARACTS_Cataract101_192pix/2022_11_09-14_54_43/'
RECYCLEGAN_ROOT = '../../results/RecycleGAN_CATARACTS_Cataract101_192pix/2022_10_13-06_59_45/'
CYCLEGAN_ROOT = '../../results/CycleGAN_CATARACTS_Cataract101_192pix/2022_10_13-06_59_50/'

SPLIT = 'Test'
ds_A = CATARACTS(
    root='/local/scratch/CATARACTS-videos-processed/',
    transforms=Tf.Compose([
        Tf.Resize((192, 192)),
        Tf.Normalize(0.5, 0.5)
    ]),
    dt=5,  # 10
    n_seq_frames=1,
    sample_phase_annotations=True,
    split=SPLIT
)

mt_unit_conf, _ = read_config(MT_UNIT_ROOT + 'config.yml')
mt_unit_conf.device = DEV

recycle_gan_conf, _ = read_config(RECYCLEGAN_ROOT + 'config.yml')
recycle_gan_conf.device = DEV

cycle_gan_conf, _ = read_config(CYCLEGAN_ROOT + 'config.yml')
cycle_gan_conf.device = DEV

mt_unit = UNIT(mt_unit_conf)
mt_unit.gen_A.load_state_dict(torch.load(MT_UNIT_ROOT + "checkpoints/gen_A_epoch79.pth", map_location='cpu'))
mt_unit.gen_A.eval()
mt_unit.gen_B.load_state_dict(torch.load(MT_UNIT_ROOT + "checkpoints/gen_B_epoch79.pth", map_location='cpu'))
mt_unit.gen_B.eval()

recycle_gan = ReCycleGAN(recycle_gan_conf)
recycle_gan.netG_A2B.load_state_dict(torch.load(RECYCLEGAN_ROOT + "checkpoints/Gen_A2B_ep190.pth", map_location='cpu'))
recycle_gan.netG_A2B.eval()

cycle_gan = CycleGAN_Agent(cycle_gan_conf)
cycle_gan.netG_A2B.load_state_dict(torch.load(CYCLEGAN_ROOT + "checkpoints/Gen_A2B_ep190.pth", map_location='cpu'))
cycle_gan.netG_A2B.eval()


case_ID = None
img_seq = None
mt_unit_translated_seq = None
recycle_gan_translated_seq = None
cycle_gan_translated_seq = None
for i, sample in enumerate(ds_A):
    frame_nr = sample['frame_nrs'][-1]
    if case_ID is None:
        case_ID = sample['case_id']
    if sample['case_id'] != case_ID or i == len(ds_A) - 1:

        # Save sequences
        sample_target_path = f"CATARACTS/{SPLIT}/{case_ID}/"
        os.makedirs(sample_target_path, exist_ok=True)

        FPS = 10

        img_seq = (denormalize(img_seq).permute(0, 2, 3, 1).cpu() * 255).to(torch.uint8)
        write_video(filename=sample_target_path + "A.mp4", video_array=img_seq, fps=FPS)

        mt_unit_translated_seq = (denormalize(mt_unit_translated_seq).permute(0, 2, 3, 1).cpu() * 255).to(torch.uint8)
        write_video(filename=sample_target_path + "mt_unit_AB.mp4", video_array=mt_unit_translated_seq, fps=FPS)

        recycle_gan_translated_seq = (denormalize(recycle_gan_translated_seq).permute(0, 2, 3, 1).cpu() * 255).to(
            torch.uint8)
        write_video(filename=sample_target_path + "recycle_gan_AB.mp4", video_array=recycle_gan_translated_seq, fps=FPS)

        cycle_gan_translated_seq = (denormalize(cycle_gan_translated_seq).permute(0, 2, 3, 1).cpu() * 255).to(
            torch.uint8)
        write_video(filename=sample_target_path + "cycle_gan_AB.mp4", video_array=cycle_gan_translated_seq, fps=FPS)

        # Reset sequences
        case_ID = sample['case_id']
        img_seq = None
        mt_unit_translated_seq = None
        recycle_gan_translated_seq = None
        cycle_gan_translated_seq = None

    img = sample['img_seq'].to(DEV)
    img_seq = img if img_seq is None else torch.cat([img_seq, img], dim=0)

    with torch.no_grad():
        h, n = mt_unit.gen_A.encode(img)
        pred = mt_unit.gen_B.decode(h + n)
        mt_unit_translated_seq = pred.cpu() if mt_unit_translated_seq is None \
            else torch.cat([mt_unit_translated_seq, pred.cpu()], dim=0)

        pred = recycle_gan.netG_A2B(img)
        recycle_gan_translated_seq = pred.cpu() if recycle_gan_translated_seq is None \
            else torch.cat([recycle_gan_translated_seq, pred.cpu()], dim=0)

        pred = cycle_gan.netG_A2B(img)
        cycle_gan_translated_seq = pred.cpu() if cycle_gan_translated_seq is None\
            else torch.cat([cycle_gan_translated_seq, pred.cpu()], dim=0)

    print(f"\rCase {case_ID} Sample {i} Frame {frame_nr}", end="")





