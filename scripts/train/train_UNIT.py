import os
import shutil
import time
from datetime import datetime

import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src.agent.unit import UNIT
from src.utils.train import read_config, compute_kl, compute_vgg_loss
from src.utils.datasets import get_dataloaders, denormalize

# Read config, make log dirs etc.
conf, uneasy_conf = read_config('src/config/UNIT_CATARACTS_cataract101_192pix.yml')
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

print("########## Loading optimizers etc.")
agent.get_opt_and_scheduler(conf)

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

    pbar = tqdm(enumerate(train_dl))
    for step, sample in pbar:

        if step == conf.training.steps_per_epoch:
            break

        real_image_A = sample["A"][:, 0].to(conf.device)
        real_image_B = sample["B"][:, 0].to(conf.device)
        fake_label = torch.full((conf.training.batch_size, 1), 0, device=conf.device, dtype=torch.float32)
        real_label = torch.full((conf.training.batch_size, 1), 0.9, device=conf.device, dtype=torch.float32)

        #
        # Discriminator update
        #

        agent.disc_opt.zero_grad()
        h_a, n_a = agent.gen_A.encode(real_image_A)
        h_b, n_b = agent.gen_B.encode(real_image_B)
        # Decode (cross domain)
        x_ba = agent.gen_A.decode(h_b + n_b)
        x_ab = agent.gen_B.decode(h_a + n_a)
        # D loss
        loss_disc_A = agent.disc_A.calc_dis_loss(x_ba.detach(), real_image_A)
        disc_A_losses.append(loss_disc_A.item())
        loss_disc_B = agent.disc_B.calc_dis_loss(x_ab.detach(), real_image_B)
        disc_B_losses.append(loss_disc_B.item())
        loss_disc_total = 1.0 * loss_disc_A + 1.0 * loss_disc_B
        loss_disc_total.backward()
        agent.disc_opt.step()

        #
        # Generator update
        #

        agent.gen_opt.zero_grad()
        # Encode
        h_a, n_a = agent.gen_A.encode(real_image_A)
        h_b, n_b = agent.gen_B.encode(real_image_B)
        # Decode (within domain)
        x_a_recon = agent.gen_A.decode(h_a + n_a)
        x_b_recon = agent.gen_B.decode(h_b + n_b)
        # Decode (cross domain)
        x_ba = agent.gen_A.decode(h_b + n_b)
        x_ab = agent.gen_B.decode(h_a + n_a)
        # Encode again
        h_b_recon, n_b_recon = agent.gen_A.encode(x_ba)
        h_a_recon, n_a_recon = agent.gen_B.encode(x_ab)
        # Decode again (if needed)
        x_aba = agent.gen_A.decode(h_a_recon + n_a_recon) if conf.model.recon_x_cyc_weight > 0 else None
        x_bab = agent.gen_B.decode(h_b_recon + n_b_recon) if conf.model.recon_x_cyc_weight > 0 else None
        # Reconstruction loss
        loss_gen_recon_x_a = torch.mean(torch.abs(x_a_recon - real_image_A))
        gen_A_recon_losses.append(loss_gen_recon_x_a.item())
        loss_gen_recon_x_b = torch.mean(torch.abs(x_b_recon - real_image_B))
        gen_B_recon_losses.append(loss_gen_recon_x_b.item())
        loss_gen_recon_kl_a = compute_kl(h_a)
        loss_gen_recon_kl_b = compute_kl(h_b)
        loss_gen_cyc_x_a = torch.mean(torch.abs(x_aba - real_image_A))
        cycle_ABA_losses.append(loss_gen_cyc_x_a.item())
        loss_gen_cyc_x_b = torch.mean(torch.abs(x_bab - real_image_B))
        cycle_BAB_losses.append(loss_gen_cyc_x_b.item())
        loss_gen_recon_kl_cyc_aba = compute_kl(h_a_recon)
        loss_gen_recon_kl_cyc_bab = compute_kl(h_b_recon)
        # GAN loss
        loss_gen_adv_a = agent.disc_A.calc_gen_loss(x_ba)
        gan_BA_losses.append(loss_gen_adv_a.item())
        loss_gen_adv_b = agent.disc_B.calc_gen_loss(x_ab)
        gan_AB_losses.append(loss_gen_adv_b.item())
        # domain-invariant perceptual loss
        loss_gen_vgg_a = compute_vgg_loss(agent.vgg, agent.instance_norm, x_ba, real_image_A) if conf.model.vgg_weight > 0 else 0
        loss_gen_vgg_b = compute_vgg_loss(agent.vgg, agent.instance_norm, x_ab, real_image_B) if conf.model.vgg_weight > 0 else 0
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
                         conf.model.vgg_weight * loss_gen_vgg_b
        loss_gen_total.backward()
        agent.gen_opt.step()

        pbar.set_description(
            f"[{epoch}/{conf.training.epochs - 1}][{step}/{len(train_dl) - 1}] "
            f"Loss_D: {loss_disc_total.item():.4f} "
            f"Loss_G: {loss_gen_total.item():.4f}"
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
        writer.add_scalar(tag='/train/LR', scalar_value=agent.lr_scheduler_gen.get_last_lr()[-1], global_step=epoch)
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

    agent.lr_scheduler_disc.step()
    agent.lr_scheduler_gen.step()
    # End of epoch
