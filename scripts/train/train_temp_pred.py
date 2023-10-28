import os
import shutil
import time
from datetime import datetime

import yaml
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.agent.recycle_unit import ReCycleUNIT
from src.utils.train import read_config
from src.utils.datasets import get_dataloaders

# Read config, make log dirs etc.
conf, uneasy_conf = read_config('src/config/recycle_UNIT_cadis_cataract.yml')
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
agent = ReCycleUNIT(conf)

print("########## Loading optimizers etc.")
agent.get_opt_and_scheduler(conf)

print("########## Training")
time.sleep(0.1)
for epoch in range(conf.training.epochs):

    temp_pred_losses = []

    pbar = tqdm(enumerate(train_dl))
    for step, sample in pbar:

        N, T, C, H, W = sample['A'].shape

        if step == conf.training.steps_per_epoch:
            break

        real_seq_A = sample["A"].to(conf.device)
        real_seq_B = sample["B"].to(conf.device)

        loss_disc_total = 0.0
        loss_gen_total = 0.0

        agent.gen_opt.zero_grad()

        # Temporal prediction loss
        next_a = agent.temp_pred_A(real_seq_A[:, :-1].view((N, (T-1)*C, H, W)))
        next_b = agent.temp_pred_B(real_seq_B[:, :-1].view((N, (T-1)*C, H, W)))
        if step == 0:
            fig, ax = plt.subplots(2, T, figsize=(3*T, 6))
            for t in range(T):
                if t < T-1:
                    ax[0, t].imshow(((real_seq_A[0, t] + 1.0)/2.0).clip(0.0, 1.0).permute(1, 2, 0).detach().cpu())
                    ax[1, t].imshow(((real_seq_B[0, t] + 1.0) / 2.0).clip(0.0, 1.0).permute(1, 2, 0).detach().cpu())
                else:
                    ax[0, t].imshow(((next_a[0] + 1.0) / 2.0).clip(0.0, 1.0).permute(1, 2, 0).detach().cpu())
                    ax[1, t].imshow(((next_b[0] + 1.0) / 2.0).clip(0.0, 1.0).permute(1, 2, 0).detach().cpu())
                ax[0, t].axis('off')
                ax[1, t].axis('off')
            plt.show()
            plt.close()
        temporal_prediction_loss = 0.5 * (torch.sqrt(F.mse_loss(next_a, real_seq_A[:, -1])) +
                                          torch.sqrt(F.mse_loss(next_b, real_seq_B[:, -1])))
        temp_pred_losses.append(temporal_prediction_loss.item())

        # Total loss
        loss_gen_total = conf.model.temp_pred_weight * temporal_prediction_loss

        loss_gen_total.backward()
        agent.gen_opt.step()

        pbar.set_description(
            f"[{epoch}/{conf.training.epochs - 1}][{step}/{len(train_dl) - 1}] "
            f"Loss_D: {loss_disc_total:.4f} "
            f"Loss_G: {loss_gen_total:.4f}"
        )

    agent.lr_scheduler_gen.step()
    # End of epoch
