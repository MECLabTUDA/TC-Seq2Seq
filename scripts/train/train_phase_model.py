import time
import os
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.inception import Inception_V3_Weights
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights
from torchvision.utils import make_grid
import torchvision.transforms as Tf
from tqdm import tqdm
from torchmetrics.classification import Accuracy, AUROC, AveragePrecision, JaccardIndex

from src.data import Cataract101
from src.model.phase_classifier_model import PhaseClassifier

print("##### Pre-train")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='Path to data root dir')
parser.add_argument('--devices', type=str, nargs='+', help='List of device names to use')
args = parser.parse_args()

DEV = args.devices[0]
T = 1
EPOCHS = 100  # 50 200
STEPS = -1  # 100, 500, -1
BATCH_SIZE = 64  # 128
NUM_WORKERS = 8
DROP_P = 0.4
VAL_FREQ = 5

LOG_DIR = 'results/resnet18_phase_model/' + datetime.now().strftime("%Y_%m_%d %H:%M:%S/")

os.makedirs(LOG_DIR, exist_ok=True)

print("##### Loading data")
train_ds = Cataract101(
    root=args.data,
    n_seq_frames=T,
    dt=1,
    transforms=Tf.Compose([
        Tf.Resize((224, 224)),
        Tf.Normalize(.5, .5)
    ]),
    split="Training",
    sample_phase_annotations=True
)
val_ds = Cataract101(
    root=args.data,
    n_seq_frames=T,
    dt=1,
    transforms=Tf.Compose([
        Tf.Resize((224, 224)),
        Tf.Normalize(.5, .5)
    ]),
    split="Validation",
    sample_phase_annotations=True
)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=False, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=False, shuffle=True)

print("##### Loading model")
m = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=ResNet18_Weights.DEFAULT).to(DEV)
for param in m.parameters():
    param.requires_grad = False
m.fc = nn.Sequential(
    nn.Dropout(p=DROP_P),
    nn.LeakyReLU(0.2),
    nn.Linear(m.fc.in_features, 256).to(DEV),
    nn.Dropout(p=DROP_P),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 11).to(DEV),
)

optim = torch.optim.AdamW(m.parameters(), lr=0.00003, weight_decay=0.001, betas=(0.9, 0.999))

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, factor=0.9, patience=5)

acc_score = Accuracy(num_classes=11).to(DEV)
ap_score = AveragePrecision(num_classes=11).to(DEV)
auroc_score = AUROC(num_classes=11).to(DEV)
f1_score = JaccardIndex(num_classes=11).to(DEV)

print("##### training phase classification model:")
time.sleep(0.1)
best_val = 0
train_losses = []
val_losses = []
val_accs = []
val_maps = []
val_f1s = []
val_aurocs = []
lr_per_epoch = []
for epoch in range(EPOCHS):

    loss_per_sample = []
    for i, sample in enumerate(tqdm(iter(train_dl))):

        if i == STEPS:
            break

        img_seq = sample['img_seq']
        phase_seq = sample['phase_seq']
        N, T, C, H, W = img_seq.shape

        optim.zero_grad()

        input = img_seq.view((N, T * C, H, W)).to(DEV)

        target = torch.argmax(phase_seq[:, -1], dim=-1).to(DEV)
        prediction = m(input)

        loss = F.cross_entropy(prediction, target)
        loss_per_sample.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 0.1)
        optim.step()

    avg_train_loss = np.mean(loss_per_sample)
    time.sleep(0.2)
    print(f"Epoch {epoch} Avg. Loss {avg_train_loss}")
    train_losses.append(avg_train_loss)
    time.sleep(0.2)

    if epoch % VAL_FREQ == 0:
        m.eval()
        with torch.no_grad():

            loss_per_sample = []
            targets = None
            predictions = None
            for i, sample in enumerate(tqdm(iter(val_dl))):

                if i == STEPS:
                    break

                img_seq = sample['img_seq']
                phase_seq = sample['phase_seq']
                N, T, C, H, W = img_seq.shape
                input = img_seq.view((N, T * C, H, W)).to(DEV)

                targets = target if targets is None \
                    else torch.cat([targets, target], dim=0)

                prediction = m(input)
                predictions = prediction if predictions is None \
                    else torch.cat([predictions, prediction], dim=0)
                loss = F.cross_entropy(prediction, target)
                loss_per_sample.append(loss.item())

            avg_val_loss = np.mean(loss_per_sample)
            acc = acc_score(predictions, targets).item()
            ap = ap_score(predictions, targets).item()
            f1 = f1_score(predictions, targets).item()
            auroc = auroc_score(predictions, targets).item()

            time.sleep(0.1)
            print(f"Avg. val. loss {avg_val_loss}\t Acc {acc}\t mAP {ap}\t F1 {f1}\t AUROC {auroc}")
            val_losses.append(avg_val_loss)
            val_accs.append(acc)
            val_maps.append(ap)
            val_f1s.append(f1)
            val_aurocs.append(auroc)
            time.sleep(0.1)

            if f1 > best_val:
                best_val = f1
                torch.save(m.state_dict(), os.path.join(LOG_DIR, "phase_model.pth"))
                print("Checkpoint saved.")

        m.train()

    # End of epoch
    lr_per_epoch.append(optim.param_groups[0]['lr'])
    scheduler.step(avg_train_loss)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(np.linspace(0, epoch, len(train_losses)), train_losses, color='blue', label='train-ce')
    ax[0].plot(np.linspace(0, epoch, len(val_losses)), val_losses, color='red', label='val-ce')
    ax[0].grid(axis='y')
    ax[0].legend()
    ax[1].plot(np.linspace(0, epoch, len(val_accs)), val_accs, color='orange', label='val-acc')
    ax[1].plot(np.linspace(0, epoch, len(val_maps)), val_maps, color='green', label='val-mAP')
    ax[1].plot(np.linspace(0, epoch, len(val_f1s)), val_f1s, color='purple', label='val-F1')
    ax[1].plot(np.linspace(0, epoch, len(val_aurocs)), val_aurocs, color='pink', label='val-AUROC')
    ax[1].grid(axis='y')
    ax[1].legend()
    ax[2].plot(np.linspace(0, epoch, len(lr_per_epoch)), lr_per_epoch, color='black', label='LR')
    ax[2].grid(axis='y')
    ax[2].legend()
    plt.autoscale()
    plt.savefig(os.path.join(LOG_DIR, "phase_model.png"))
    plt.close()
