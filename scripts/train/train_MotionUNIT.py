import os
import shutil
import time
from datetime import datetime

import yaml
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src.agent.unit import UNIT
from src.model.unet import UNet
from src.utils.train import read_config, compute_kl, compute_vgg_loss
from src.utils.datasets import get_dataloaders, denormalize

# Read config, make log dirs etc.
conf, uneasy_conf = read_config('src/config/MotionUNIT_CATARACTS_cataract101_192pix.yml')
date_time_string = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
OUT_DIR = conf.log_dir + date_time_string + "/"
if os.path.exists(OUT_DIR) and os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR + "samples/")
os.makedirs(OUT_DIR + "checkpoints/")
with open(OUT_DIR + "config.yml", 'w') as conf_file:
    yaml.dump(uneasy_conf, conf_file)
writer = SummaryWriter(log_dir=OUT_DIR)
writer.add_text(tag='config', text_string=yaml.dump(uneasy_conf, default_flow_style=False))

print("########## Loading data.")
train_dl, test_dl = get_dataloaders(conf)

print("########## Loading model.")
agent = UNIT(conf)
weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()
flo_model = raft_large(weights=weights, progress=False).to(conf.device)
flo_model = flo_model.eval()
motion_translator_AB = UNet(n_channels=3+3+3+2, n_classes=2).to(conf.device)
motion_translator_AB.load_state_dict(torch.load(
'results/MotionUNIT_CATARACTS_Cataract101_192pix/2022_11_03-12_18_46/checkpoints/mt_AB_epoch139.pth'
))
motion_translator_BA = UNet(n_channels=3+3+3+2, n_classes=2).to(conf.device)
motion_translator_BA.load_state_dict(torch.load(
'results/MotionUNIT_CATARACTS_Cataract101_192pix/2022_11_03-12_18_46/checkpoints/mt_BA_epoch139.pth'
))
MSSIM = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(conf.device)

print("########## Loading optimizers etc.")
agent.get_opt_and_scheduler(conf)
agent.load_from_checkpoint('results/MotionUNIT_CATARACTS_Cataract101_192pix/2022_11_03-12_18_46/checkpoints/')

print("########## Training")
time.sleep(0.1)

for epoch in range(conf.training.epochs):

    disc_A_losses = []
    disc_B_losses = []
    gen_A_recon_losses = []
    gen_B_recon_losses = []
    cycle_ABA_losses = []
    cycle_BAB_losses = []
    gan_AB_losses = []
    gan_BA_losses = []
    perceptual_losses_A = []
    perceptual_losses_B = []
    mt_losses_AB = []
    mt_losses_BA = []
    ms_ssim_losses_AB = []
    ms_ssim_losses_BA = []

    pbar = tqdm(enumerate(train_dl))
    for step, sample in pbar:

        if step == conf.training.steps_per_epoch:
            break

        seq_A = sample["A"].to(conf.device)
        seq_B = sample["B"].to(conf.device)
        N, T, C, H, W = seq_A.shape
        fake_label = torch.full((conf.training.batch_size, 1), 0, device=conf.device, dtype=torch.float32)
        real_label = torch.full((conf.training.batch_size, 1), 0.9, device=conf.device, dtype=torch.float32)

        #
        # Discriminator update
        #

        agent.disc_opt.zero_grad()
        h_a, n_a = agent.gen_A.encode(seq_A[:, -1])
        h_b, n_b = agent.gen_B.encode(seq_B[:, -1])
        # Decode (cross domain)
        x_ba = agent.gen_A.decode(h_b + n_b)
        x_ab = agent.gen_B.decode(h_a + n_a)
        # D loss
        loss_disc_A = agent.disc_A.calc_dis_loss(x_ba.detach(), seq_A[:, -1])
        disc_A_losses.append(loss_disc_A.item())
        loss_disc_B = agent.disc_B.calc_dis_loss(x_ab.detach(), seq_B[:, -1])
        disc_B_losses.append(loss_disc_B.item())
        loss_disc_total = conf.model.disc.l_weight_A * loss_disc_A + conf.model.disc.l_weight_A * loss_disc_B
        loss_disc_total.backward()
        agent.disc_opt.step()

        #
        # Generator update
        #

        agent.gen_opt.zero_grad()
        # Encode
        h_a_2, n_a_2 = agent.gen_A.encode(seq_A[:, -1])
        h_b_2, n_b_2 = agent.gen_B.encode(seq_B[:, -1])
        # Decode (within domain)
        x_a_2_recon = agent.gen_A.decode(h_a_2 + n_a_2)
        x_b_2_recon = agent.gen_B.decode(h_b_2 + n_b_2)
        # Decode (cross domain)
        x_ba_2 = agent.gen_A.decode(h_b_2 + n_b_2)
        x_ab_2 = agent.gen_B.decode(h_a_2 + n_a_2)
        # Encode again
        h_b_2_recon, n_b_2_recon = agent.gen_A.encode(x_ba_2)
        h_a_2_recon, n_a_2_recon = agent.gen_B.encode(x_ab_2)
        # Decode again (if needed)
        x_aba_2 = agent.gen_A.decode(h_a_2_recon + n_a_2_recon) if conf.model.recon_x_cyc_weight > 0 else None
        x_bab_2 = agent.gen_B.decode(h_b_2_recon + n_b_2_recon) if conf.model.recon_x_cyc_weight > 0 else None
        # Reconstruction loss
        loss_gen_recon_x_a = torch.mean(torch.abs(x_a_2_recon - seq_A[:, -1]))
        gen_A_recon_losses.append(loss_gen_recon_x_a.item())
        loss_gen_recon_x_b = torch.mean(torch.abs(x_b_2_recon - seq_B[:, -1]))
        gen_B_recon_losses.append(loss_gen_recon_x_b.item())
        loss_gen_recon_kl_a = compute_kl(h_a_2)
        loss_gen_recon_kl_b = compute_kl(h_b_2)
        loss_gen_cyc_x_a = torch.mean(torch.abs(x_aba_2 - seq_A[:, -1]))
        cycle_ABA_losses.append(loss_gen_cyc_x_a.item())
        loss_gen_cyc_x_b = torch.mean(torch.abs(x_bab_2 - seq_B[:, -1]))
        cycle_BAB_losses.append(loss_gen_cyc_x_b.item())
        loss_gen_recon_kl_cyc_aba = compute_kl(h_a_2_recon)
        loss_gen_recon_kl_cyc_bab = compute_kl(h_b_2_recon)
        # GAN loss
        loss_gen_adv_a = agent.disc_A.calc_gen_loss(x_ba_2)
        gan_BA_losses.append(loss_gen_adv_a.item())
        loss_gen_adv_b = agent.disc_B.calc_gen_loss(x_ab_2)
        gan_AB_losses.append(loss_gen_adv_b.item())
        # domain-invariant perceptual loss
        loss_gen_vgg_a = compute_vgg_loss(agent.vgg, agent.instance_norm, x_ba_2, seq_A[:, -1]) if conf.model.vgg_weight > 0 else 0
        loss_gen_vgg_b = compute_vgg_loss(agent.vgg, agent.instance_norm, x_ab_2, seq_B[:, -1]) if conf.model.vgg_weight > 0 else 0
        perceptual_losses_A.append(loss_gen_vgg_a.item())
        perceptual_losses_B.append(loss_gen_vgg_b.item())

        #
        # Motion translation loss
        #

        if conf.model.mt_weight > 0 or conf.model.msssim_weight > 0:
            # Translated previous frames
            # Encode
            h_a_1, n_a_1 = agent.gen_A.encode(seq_A[:, -2])
            h_b_1, n_b_1 = agent.gen_B.encode(seq_B[:, -2])
            # Decode (cross domain)
            x_ba_1 = agent.gen_A.decode(h_b_1 + n_b_1)
            x_ab_1 = agent.gen_B.decode(h_a_1 + n_a_1)

        if conf.model.mt_weight > 0:
            # Estimate flow between source frames
            with torch.no_grad():
                a_flow = flo_model(seq_A[:, -2], seq_A[:, -1])[-1]
                b_flow = flo_model(seq_B[:, -2], seq_B[:, -2])[-1]

            # Translate flow using motion translation model
            translated_a_flow = motion_translator_AB(
                torch.cat([seq_A.view((N, T * C, H, W)), x_ab_1.detach(), a_flow], dim=1))
            translated_b_flow = motion_translator_BA(
                torch.cat([seq_B.view((N, T * C, H, W)), x_ba_1.detach(), b_flow], dim=1))

            # Warp target domain frames to previous frames
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(N, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(N, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float().to(conf.device)

            a_vgrid = Variable(grid) - translated_b_flow
            a_vgrid[:, 0, :, :] = 2.0 * a_vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
            a_vgrid[:, 1, :, :] = 2.0 * a_vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
            a_vgrid = a_vgrid.permute(0, 2, 3, 1)

            b_vgrid = Variable(grid) - translated_a_flow
            b_vgrid[:, 0, :, :] = 2.0 * b_vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
            b_vgrid[:, 1, :, :] = 2.0 * b_vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
            b_vgrid = b_vgrid.permute(0, 2, 3, 1)

            x_ba_2_warped = F.grid_sample(x_ba_1, a_vgrid, align_corners=True)
            a_mask = torch.autograd.Variable(torch.ones(x_ba_1.size())).to(conf.device)
            a_mask = F.grid_sample(a_mask, a_vgrid)
            a_mask[a_mask < 0.9999] = 0
            a_mask[a_mask > 0] = 1
            a_2_warped = a_mask * x_ba_2_warped

            x_ab_2_warped = F.grid_sample(x_ab_1, b_vgrid, align_corners=True)
            b_mask = torch.autograd.Variable(torch.ones(x_ab_1.size())).to(conf.device)
            b_mask = F.grid_sample(b_mask, b_vgrid)
            b_mask[b_mask < 0.9999] = 0
            b_mask[b_mask > 0] = 1
            b_2_warped = b_mask * x_ab_2_warped

            mt_loss_AB = F.mse_loss(x_ab_2.detach(), x_ab_2_warped)
            mt_loss_BA = F.mse_loss(x_ba_2.detach(), x_ba_2_warped)
        else:
            mt_loss_AB = mt_loss_BA = torch.tensor([0]).to(conf.device)

        mt_losses_AB.append(mt_loss_AB.item())
        mt_losses_BA.append(mt_loss_BA.item())

        # Cross-time MS-SSIM loss
        ms_ssim_loss_BA = -MSSIM(x_ba_2_warped.detach(), x_ba_1) if conf.model.msssim_weight > 0 else torch.tensor([0]).to(conf.device)
        ms_ssim_loss_AB = -MSSIM(x_ab_2_warped.detach(), x_ab_1) if conf.model.msssim_weight > 0 else torch.tensor([0]).to(conf.device)
        ms_ssim_losses_BA.append(ms_ssim_loss_BA.item())
        ms_ssim_losses_AB.append(ms_ssim_loss_AB.item())

        # total loss
        loss_gen_total = conf.model.gan_weight * loss_gen_adv_a + \
                         conf.model.gan_weight * loss_gen_adv_b + \
                         conf.model.recon_x_weight * loss_gen_recon_x_a + \
                         conf.model.recon_x_weight * loss_gen_recon_x_b + \
                         conf.model.kl_weight * loss_gen_recon_kl_a + \
                         conf.model.kl_weight * loss_gen_recon_kl_b + \
                         conf.model.recon_x_cyc_weight * loss_gen_cyc_x_a + \
                         conf.model.recon_x_cyc_weight * loss_gen_cyc_x_b + \
                         conf.model.recon_kl_cyc_weight * loss_gen_recon_kl_cyc_aba + \
                         conf.model.recon_kl_cyc_weight * loss_gen_recon_kl_cyc_bab + \
                         conf.model.vgg_weight * loss_gen_vgg_a + \
                         conf.model.vgg_weight * loss_gen_vgg_b + \
                         conf.model.mt_weight * mt_loss_AB + \
                         conf.model.mt_weight * mt_loss_BA + \
                         conf.model.msssim_weight * ms_ssim_loss_AB + \
                         conf.model.msssim_weight * ms_ssim_loss_BA
        loss_gen_total.backward()
        agent.gen_opt.step()

        pbar.set_description(
            f"[{epoch}/{conf.training.epochs - 1}][{step}/{len(train_dl) - 1}] "
            f"Loss_D: {loss_disc_total.item():.4f} "
            f"Loss_G: {loss_gen_total.item():.4f} "
            f"Loss_OF: {0.5*(mt_loss_AB + mt_loss_BA).item():.4f}"
        )

    with torch.no_grad():
        # Dump training values to tensorboard
        writer.add_scalar(tag='train/D_A', scalar_value=np.mean(disc_A_losses), global_step=epoch)
        writer.add_scalar(tag='train/D_B', scalar_value=np.mean(disc_B_losses), global_step=epoch)
        writer.add_scalar(tag='train/G_Id_A', scalar_value=np.mean(gen_A_recon_losses), global_step=epoch)
        writer.add_scalar(tag='train/G_Id_B', scalar_value=np.mean(gen_B_recon_losses), global_step=epoch)
        writer.add_scalar(tag='train/G_ABA_cycle', scalar_value=np.mean(cycle_ABA_losses), global_step=epoch)
        writer.add_scalar(tag='train/G_BAB_cycle', scalar_value=np.mean(cycle_BAB_losses), global_step=epoch)
        writer.add_scalar(tag='train/G_AB', scalar_value=np.mean(gan_AB_losses), global_step=epoch)
        writer.add_scalar(tag='train/G_BA', scalar_value=np.mean(gan_BA_losses), global_step=epoch)
        writer.add_scalar(tag='train/VGG_A', scalar_value=np.mean(perceptual_losses_A), global_step=epoch)
        writer.add_scalar(tag='train/VGG_B', scalar_value=np.mean(perceptual_losses_B), global_step=epoch)
        writer.add_scalar(tag='train/MT_AB', scalar_value=np.mean(mt_losses_AB), global_step=epoch)
        writer.add_scalar(tag='train/MT_BA', scalar_value=np.mean(mt_losses_AB), global_step=epoch)
        writer.add_scalar(tag='train/MS_SSIM_AB', scalar_value=np.mean(ms_ssim_losses_AB), global_step=epoch)
        writer.add_scalar(tag='train/MS_SSIM_BA', scalar_value=np.mean(ms_ssim_losses_BA), global_step=epoch)
        writer.add_scalar(tag='train/LR', scalar_value=agent.lr_scheduler_gen.get_last_lr()[-1], global_step=epoch)
        writer.flush()

    if not (epoch % conf.validation.save_samples_freq):
        with torch.no_grad():

            sample = next(iter(test_dl))
            real_seq_A = sample["A"].to(conf.device)
            real_seq_B = sample["B"].to(conf.device)
            N, T, C, H, W = real_seq_A.shape

            for n in range(N):

                fig, ax = plt.subplots(6, T, figsize=(T * 3, 18))

                for t in range(T):
                    h_a, _ = agent.gen_A.encode(real_seq_A[n, t:t+1])
                    h_b, _ = agent.gen_B.encode(real_seq_B[n, t:t+1])

                    x_a_recon = agent.gen_A.decode(h_a)
                    x_b_recon = agent.gen_B.decode(h_b)
                    x_ba = agent.gen_A.decode(h_b)
                    x_ab = agent.gen_B.decode(h_a)

                    ax[0, t].imshow(denormalize(real_seq_A[n, t]).permute(1, 2, 0).cpu().numpy())
                    ax[1, t].imshow(denormalize(x_a_recon[0]).permute(1, 2, 0).cpu().numpy())
                    ax[2, t].imshow(denormalize(x_ab[0]).permute(1, 2, 0).cpu().numpy())
                    ax[3, t].imshow(denormalize(real_seq_B[n, t]).permute(1, 2, 0).cpu().numpy())
                    ax[4, t].imshow(denormalize(x_b_recon[0]).permute(1, 2, 0).cpu().numpy())
                    ax[5, t].imshow(denormalize(x_ba[0]).permute(1, 2, 0).cpu().numpy())

                plt.savefig(OUT_DIR + f"samples/epoch{epoch}_sample{n}.SVG")
                plt.close()

    if not ((epoch + 1) % conf.validation.save_checkpoint_freq):
        torch.save(agent.gen_A.state_dict(), f=OUT_DIR + f"checkpoints/gen_A_epoch{epoch}.pth")
        torch.save(agent.gen_B.state_dict(), f=OUT_DIR + f"checkpoints/gen_B_epoch{epoch}.pth")
        torch.save(agent.disc_A.state_dict(), f=OUT_DIR + f"checkpoints/disc_A_epoch{epoch}.pth")
        torch.save(agent.disc_B.state_dict(), f=OUT_DIR + f"checkpoints/disc_B_epoch{epoch}.pth")
        torch.save(agent.gen_opt.state_dict(), f=OUT_DIR + f"checkpoints/gen_opt_epoch{epoch}.pth")
        torch.save(agent.disc_opt.state_dict(), f=OUT_DIR + f"checkpoints/disc_opt_epoch{epoch}.pth")
        torch.save(motion_translator_AB.state_dict(), f=OUT_DIR + f"checkpoints/mt_AB_epoch{epoch}.pth")
        torch.save(motion_translator_BA.state_dict(), f=OUT_DIR + f"checkpoints/mt_BA_epoch{epoch}.pth")

    agent.lr_scheduler_disc.step()
    agent.lr_scheduler_gen.step()
    # End of epoch
