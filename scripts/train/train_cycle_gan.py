import os
import shutil
from datetime import datetime

import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.agent.cycle_gan import CycleGAN_Agent
from src.utils.datasets import get_dataloaders, denormalize
from src.utils.model import ReplayBuffer
from src.utils.train import read_config

# Parameters etc.
conf, uneasy_conf = read_config('src/config/cyclegan_CATARACTS_cataract101_192pix.yml')
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
agent = CycleGAN_Agent(conf)

print("########## Loading optimizers etc.")
agent.get_opt_and_scheduler(conf)

print("########## Training")

# Loss functions
cycle_loss = nn.L1Loss().to(conf.device)
identity_loss = nn.L1Loss().to(conf.device)
adversarial_loss = nn.MSELoss().to(conf.device)  # Least-Squares replacing NLL for more stability

# Image buffers to update the discriminators
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

for epoch in range(conf.training.epochs):

    g_losses = []
    d_losses = []

    identity_losses = []
    gan_losses = []
    cycle_losses = []

    pbar = tqdm(enumerate(train_dl))
    for step, sample in pbar:

        if step == conf.training.steps_per_epoch:
            break

        real_image_A = sample["A"][:, 0].to(conf.device)
        real_image_B = sample["B"][:, 0].to(conf.device)
        fake_label = torch.full((conf.training.batch_size, 1), 0, device=conf.device, dtype=torch.float32)
        # real_label = torch.full((conf.training.batch_size, 1), 1, device=conf.device, dtype=torch.float32)
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
        identity_image_A = agent.netG_B2A(real_image_A)
        loss_identity_A = identity_loss(identity_image_A, real_image_A) * conf.training.identity_loss_weight
        # G_A2B(B) should equal B if real B is fed
        identity_image_B = agent.netG_A2B(real_image_B)
        loss_identity_B = identity_loss(identity_image_B, real_image_B) * conf.training.identity_loss_weight
        identity_losses.append((loss_identity_A.item(), loss_identity_B.item()))

        # GAN loss
        # GAN loss D_A(G_A(A))
        fake_image_A = agent.netG_B2A(real_image_B)
        fake_output_A = agent.netD_A(fake_image_A)
        loss_GAN_B2A = adversarial_loss(fake_output_A, real_label) * conf.training.adversarial_loss_weight
        # GAN loss D_B(G_B(B))
        fake_image_B = agent.netG_A2B(real_image_A)
        fake_output_B = agent.netD_B(fake_image_B)
        loss_GAN_A2B = adversarial_loss(fake_output_B, real_label) * conf.training.adversarial_loss_weight
        gan_losses.append((loss_GAN_A2B.item(), loss_GAN_B2A.item()))

        # Cycle loss
        recovered_image_A = agent.netG_B2A(fake_image_B)
        loss_cycle_ABA = cycle_loss(recovered_image_A, real_image_A) * conf.training.cycle_loss_weight

        recovered_image_B = agent.netG_A2B(fake_image_A)
        loss_cycle_BAB = cycle_loss(recovered_image_B, real_image_B) * conf.training.cycle_loss_weight
        cycle_losses.append((loss_cycle_ABA.item(), loss_cycle_BAB.item()))

        # Combined loss and calculate gradients
        errG = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
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
        real_output_A = agent.netD_A(real_image_A)
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
        real_output_B = agent.netD_B(real_image_B)
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
            f"Loss_D: {np.mean(d_losses):.4f} "
            f"Loss_G: {np.mean(gan_losses):.4f} "
            f"Loss_G_identity: {np.mean(identity_losses):.4f} "
            f"loss_G_GAN: {np.mean(gan_losses):.4f} "
            f"loss_G_cycle: {np.mean(cycle_losses):.4f}")

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
                    fake_image_A = agent.netG_B2A(real_seq_B[n, t])
                    fake_image_B = agent.netG_A2B(real_seq_A[n, t])

                    recon_image_A = agent.netG_B2A(real_seq_A[n, t])
                    recon_image_B = agent.netG_A2B(real_seq_B[n, t])

                    real_image_A = real_seq_A[n, t]
                    real_image_B = real_seq_B[n, t]

                    ax[0, t].imshow(denormalize(real_image_A).permute(1, 2, 0).cpu().numpy())
                    ax[1, t].imshow(denormalize(recon_image_A).permute(1, 2, 0).cpu().numpy())
                    ax[2, t].imshow(denormalize(fake_image_B).permute(1, 2, 0).cpu().numpy())
                    ax[3, t].imshow(denormalize(real_image_B).permute(1, 2, 0).cpu().numpy())
                    ax[4, t].imshow(denormalize(recon_image_B).permute(1, 2, 0).cpu().numpy())
                    ax[5, t].imshow(denormalize(fake_image_A).permute(1, 2, 0).cpu().numpy())

                plt.savefig(OUT_DIR + f"samples/epoch{epoch}_sample{n}.SVG")
                plt.close()

    if epoch % conf.validation.save_freq_checkpoints == 0:
        torch.save(agent.netG_A2B.state_dict(), OUT_DIR + f"checkpoints/Gen_A2B_ep{epoch}.pth")
        torch.save(agent.netG_B2A.state_dict(), OUT_DIR + f"checkpoints/Gen_B2A_ep{epoch}.pth")
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
    d_losses = np.mean(d_losses, axis=0)
    writer.add_scalar(tag='/train/D_A', scalar_value=d_losses[0], global_step=epoch)
    writer.add_scalar(tag='/train/D_B', scalar_value=d_losses[1], global_step=epoch)
    writer.add_scalar(tag='/train/LR', scalar_value=agent.lr_scheduler_G.get_last_lr()[-1], global_step=epoch)
    writer.flush()

    # Update learning rates after each epoch
    agent.lr_scheduler_G.step()
    agent.lr_scheduler_D_A.step()
    agent.lr_scheduler_D_B.step()
