import os
import shutil
from datetime import datetime

import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.agent.recycle_gan import ReCycleGAN
from src.utils.datasets import get_dataloaders
from src.utils.model import ReplayBuffer
from src.utils.train import read_config

# Parameters etc.
conf, uneasy_conf = read_config('src/config/recycle_gan_CATARACTS_cataract101_192pix.yml')
date_time_string = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
OUT_DIR = conf.log_dir + date_time_string + "/"
if os.path.exists(OUT_DIR) and os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR + "samples/")
os.makedirs(OUT_DIR + "checkpoints/")
with open(OUT_DIR + "config.yml", 'w') as conf_file:
    yaml.dump(uneasy_conf, conf_file)
writer = SummaryWriter(log_dir=OUT_DIR)
writer.add_text(tag='config', text_string=json.dumps(uneasy_conf, indent=4))

print("########## Loading data.")
train_dl, test_dl = get_dataloaders(conf)

print("########## Loading model.")
agent = ReCycleGAN(conf)

print("########## Loading optimizers etc.")
agent.get_opt_and_scheduler(conf)

print("########## Training")
# Loss functions
cycle_loss = nn.L1Loss().to(conf.device)
identity_loss = nn.L1Loss().to(conf.device)
adversarial_loss = nn.MSELoss().to(conf.device)  # Leas-Squares replacing NLL for more stability
recurrent_loss = nn.MSELoss().to(conf.device)
recycle_loss = nn.MSELoss().to(conf.device)

# Image buffers to update the discriminators
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

for epoch in range(conf.training.epochs):

    g_losses = []
    d_losses = []

    identity_losses = []
    gan_losses = []
    cycle_losses = []
    temp_pred_losses = []
    recycle_losses = []

    pbar = tqdm(enumerate(train_dl))
    for step, sample in pbar:

        if step == conf.training.steps_per_epoch:
            break

        real_seq_A = sample["A"].to(conf.device)
        real_seq_B = sample["B"].to(conf.device)
        N, T, C, H, W = real_seq_A.shape
        fake_label = torch.full((conf.training.batch_size, 1), 0, device=conf.device, dtype=torch.float32)
        # One-sided label smoothing
        real_label = torch.full((conf.training.batch_size, 1), 0.9, device=conf.device, dtype=torch.float32)

        #
        #
        # Updating generator networks
        #
        #

        # Set G_A and G_B's gradients to zero
        agent.optimizer_G.zero_grad()

        # Identity loss
        # G_B2A(A) should equal A if real A is fed
        identity_image_A = agent.netG_B2A(real_seq_A[:, -1])
        loss_identity_A = identity_loss(identity_image_A, real_seq_A[:, -1]) * conf.training.identity_loss_weight
        # G_A2B(B) should equal B if real B is fed
        identity_image_B = agent.netG_A2B(real_seq_B[:, -1])
        loss_identity_B = identity_loss(identity_image_B, real_seq_B[:, -1]) * conf.training.identity_loss_weight
        identity_losses.append((loss_identity_A.item(), loss_identity_B.item()))

        # GAN loss
        # GAN loss D_A(G_A(A))
        fake_image_A = agent.netG_B2A(real_seq_B[:, -1])
        fake_output_A = agent.netD_A(fake_image_A)
        loss_GAN_B2A = adversarial_loss(fake_output_A, real_label) * conf.training.adversarial_loss_weight
        # GAN loss D_B(G_B(B))
        fake_image_B = agent.netG_A2B(real_seq_A[:, -1])
        fake_output_B = agent.netD_B(fake_image_B)
        loss_GAN_A2B = adversarial_loss(fake_output_B, real_label) * conf.training.adversarial_loss_weight
        gan_losses.append((loss_GAN_A2B.item(), loss_GAN_B2A.item()))

        # Cycle loss
        recovered_image_A = agent.netG_B2A(fake_image_B)
        loss_cycle_ABA = cycle_loss(recovered_image_A, real_seq_A[:, -1]) * conf.training.cycle_loss_weight

        recovered_image_B = agent.netG_A2B(fake_image_A)
        loss_cycle_BAB = cycle_loss(recovered_image_B, real_seq_A[:, -1]) * conf.training.cycle_loss_weight
        cycle_losses.append((loss_cycle_ABA.item(), loss_cycle_BAB.item()))

        # Temporal prediction loss
        predicted_image_A = agent.temp_pred_A(real_seq_A[:, :-1].view((N, (T-1)*C, H, W)))
        loss_temp_pred_A = recurrent_loss(predicted_image_A, real_seq_A[:, -1]) * conf.training.recurrent_loss_weight
        predicted_image_B = agent.temp_pred_B(real_seq_B[:, :-1].view((N, (T - 1) * C, H, W)))
        loss_temp_pred_B = recurrent_loss(predicted_image_B, real_seq_B[:, -1]) * conf.training.recurrent_loss_weight
        temp_pred_losses.append((loss_temp_pred_A.item(), loss_temp_pred_B.item()))

        # Recycle loss
        recycled_image_A = agent.netG_A2B(real_seq_A[:, :-1].reshape((N * (T - 1), C, H, W)))
        recycled_image_A = agent.temp_pred_B(recycled_image_A.view((N, (T - 1)*C, H, W)))
        recycled_image_A = agent.netG_B2A(recycled_image_A.view((N, C, H, W)))
        loss_recycle_A = recycle_loss(recycled_image_A, real_seq_A[:, -1])* conf.training.recycle_loss_weight
        recycled_image_B = agent.netG_B2A(real_seq_B[:, :-1].reshape((N * (T - 1), C, H, W)))
        recycled_image_B = agent.temp_pred_A(recycled_image_B.view((N, (T - 1) * C, H, W)))
        recycled_image_B = agent.netG_A2B(recycled_image_B.view((N, C, H, W)))
        loss_recycle_B = recycle_loss(recycled_image_B, real_seq_B[:, -1]) * conf.training.recycle_loss_weight
        recycle_losses.append((loss_recycle_A.item(), loss_recycle_B.item()))

        # Combined loss and calculate gradients
        errG = loss_identity_A + loss_identity_B +\
               loss_GAN_A2B + loss_GAN_B2A +\
               loss_cycle_ABA + loss_cycle_BAB +\
               loss_temp_pred_A + loss_temp_pred_B +\
               loss_recycle_A + loss_recycle_B
        g_losses.append(errG.item())

        # Calculate gradients for G_A and G_B
        errG.backward()
        # Update G_A and G_B's weights
        agent.optimizer_G.step()

        #
        #
        # Update discriminator A
        #
        #

        # Set D_A gradients to zero
        agent.optimizer_D_A.zero_grad()

        # Real A image loss
        real_output_A = agent.netD_A(real_seq_A[:, -1])
        errD_real_A = adversarial_loss(real_output_A, real_label)

        # Fake A image loss
        fake_image_A = fake_A_buffer.push_and_pop(fake_image_A)
        fake_output_A = agent.netD_A(fake_image_A.detach())
        errD_fake_A = adversarial_loss(fake_output_A, fake_label)

        # Combined loss and calculate gradients
        errD_A = 0.5 * (errD_real_A + errD_fake_A) * conf.training.discriminator_loss_weight

        # Calculate gradients for D_A
        errD_A.backward()
        # Update D_A weights
        agent.optimizer_D_A.step()

        #
        #
        # Update discriminator B
        #
        #

        # Set D_B gradients to zero
        agent.optimizer_D_B.zero_grad()

        # Real B image loss
        real_output_B = agent.netD_B(real_seq_B[:, -1])
        errD_real_B = adversarial_loss(real_output_B, real_label)

        # Fake B image loss
        fake_image_B = fake_B_buffer.push_and_pop(fake_image_B)
        fake_output_B = agent.netD_B(fake_image_B.detach())
        errD_fake_B = adversarial_loss(fake_output_B, fake_label)

        # Combined loss and calculate gradients
        errD_B = 0.5 * (errD_real_B + errD_fake_B) * conf.training.discriminator_loss_weight

        # Calculate gradients for D_B
        errD_B.backward()
        # Update D_B weights
        agent.optimizer_D_B.step()

        d_losses.append((errD_A.item(), errD_B.item()))

        pbar.set_description(
            f"[{epoch}/{conf.training.epochs - 1}][{step}/{len(train_dl) - 1}] "
            f"L_D: {np.mean(d_losses):.3f} "
            f"L_G: {np.mean(g_losses):.3f} "
            f"L_G_iden: {np.mean(identity_losses):.3f} "
            f"L_G_GAN: {np.mean(gan_losses):.3f} "
            f"L_G_cycle: {np.mean(cycle_losses):.3f} "
            f"L_temp: {np.mean(temp_pred_losses):.3f} "
            f"L_recycle: {np.mean(recycle_losses):.3f}")

    # Save some results every couple of epochs
    if epoch % conf.validation.save_freq_samples == 0:

        with torch.no_grad():

            sample = next(iter(test_dl))
            real_seq_A = sample["A"].to(conf.device)
            real_seq_B = sample["B"].to(conf.device)
            N, T, C, H, W = real_seq_A.shape

            for n in range(N):

                fig, ax = plt.subplots(6, T, figsize=(T * 3, 18))

                for t in range(T):

                    fake_image_A = 0.5 * (agent.netG_B2A(real_seq_B[n, t]).data + 1.0)
                    fake_image_B = 0.5 * (agent.netG_A2B(real_seq_A[n, t]).data + 1.0)

                    recon_image_A = 0.5 * (agent.netG_B2A(real_seq_A[n, t]).data + 1.0)
                    recon_image_B = 0.5 * (agent.netG_A2B(real_seq_B[n, t]).data + 1.0)

                    real_image_A = 0.5 * (real_seq_A[n, t] + 1.0)
                    real_image_B = 0.5 * (real_seq_B[n, t] + 1.0)

                    ax[0, t].imshow(real_image_A.permute(1, 2, 0).cpu().numpy())
                    ax[1, t].imshow(recon_image_A.permute(1, 2, 0).cpu().numpy())
                    ax[2, t].imshow(fake_image_B.permute(1, 2, 0).cpu().numpy())
                    ax[3, t].imshow(real_image_B.permute(1, 2, 0).cpu().numpy())
                    ax[4, t].imshow(recon_image_B.permute(1, 2, 0).cpu().numpy())
                    ax[5, t].imshow(fake_image_A.permute(1, 2, 0).cpu().numpy())

                plt.savefig(OUT_DIR + f"samples/epoch{epoch}_sample{n}.SVG")
                plt.close()

    if epoch % conf.validation.save_freq_checkpoints == 0:
        torch.save(agent.netG_A2B.state_dict(), OUT_DIR + f"checkpoints/Gen_A2B_ep{epoch}.pth")
        torch.save(agent.netG_B2A.state_dict(), OUT_DIR + f"checkpoints/Gen_B2A_ep{epoch}.pth")
        torch.save(agent.temp_pred_A.state_dict(), OUT_DIR + f"checkpoints/Temp_A_ep{epoch}.pth")
        torch.save(agent.temp_pred_B.state_dict(), OUT_DIR + f"checkpoints/Temp_B_ep{epoch}.pth")
        torch.save(agent.netD_A.state_dict(), OUT_DIR + f"checkpoints/Disc_A_ep{epoch}.pth")
        torch.save(agent.netD_B.state_dict(), OUT_DIR + f"checkpoints/Disc_B_ep{epoch}.pth")

    # Write losses to tensorboard
    identity_losses = np.mean(identity_losses, axis=0)
    writer.add_scalar(tag='/train/Identity_A', scalar_value=identity_losses[0], global_step=epoch)
    writer.add_scalar(tag='/train/Identity_B', scalar_value=identity_losses[1], global_step=epoch)
    gan_losses = np.mean(gan_losses, axis=0)
    writer.add_scalar(tag='/train/GAN_A2B', scalar_value=gan_losses[0], global_step=epoch)
    writer.add_scalar(tag='/train/GAN_B2A', scalar_value=gan_losses[1], global_step=epoch)
    cycle_losses = np.mean(cycle_losses, axis=0)
    writer.add_scalar(tag='/train/Cycle_ABA', scalar_value=cycle_losses[0], global_step=epoch)
    writer.add_scalar(tag='/train/Cycle_BAB', scalar_value=cycle_losses[1], global_step=epoch)
    temp_pred_losses = np.mean(temp_pred_losses, axis=0)
    writer.add_scalar(tag='/train/Temp_A', scalar_value=temp_pred_losses[0], global_step=epoch)
    writer.add_scalar(tag='/train/Temp_B', scalar_value=temp_pred_losses[1], global_step=epoch)
    reycle_losses = np.mean(recycle_losses, axis=0)
    writer.add_scalar(tag='/train/reycle_ABA', scalar_value=reycle_losses[0], global_step=epoch)
    writer.add_scalar(tag='/train/reycle_BAB', scalar_value=reycle_losses[1], global_step=epoch)
    d_losses = np.mean(d_losses, axis=0)
    writer.add_scalar(tag='/train/D_A', scalar_value=d_losses[0], global_step=epoch)
    writer.add_scalar(tag='/train/D_B', scalar_value=d_losses[1], global_step=epoch)
    writer.add_scalar(tag='/train/LR', scalar_value=agent.lr_scheduler_G.get_last_lr()[-1], global_step=epoch)
    writer.flush()

    # Update learning rates after each epoch
    agent.lr_scheduler_G.step()
    agent.lr_scheduler_D_A.step()
    agent.lr_scheduler_D_B.step()
