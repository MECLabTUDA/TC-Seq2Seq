import time
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as Tf
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout
from tqdm import tqdm

from src.data import Cataract101, CATARACTS
from src.model.phase_classifier_model import PhaseClassifier
from src.utils.train import read_config, DecayLR, get_lr

dev = 'cuda:0'

T = 3
EPOCHS = 600  # 200
STEPS = 500  # 1000
BATCH_SIZE = 16  # 64  # 16
NUM_WORKERS = 8
VAL_FREQ = 5
DIR = 'results/phase_model/phase_model_reduce_lr_on_plateau4/'
os.makedirs(DIR, exist_ok=True)

print("##### Loading data")


def identity_(t: torch.Tensor):
    pass


train_ds_1 = Cataract101(
    root='/local/scratch/cataract-101-processed/',
    n_seq_frames=T,
    dt=1,
    transforms=Tf.Compose([
        Tf.Resize((128, 128)),
        Tf.RandomChoice([
            Tf.RandomHorizontalFlip(),
            Tf.RandomVerticalFlip(),
        ], p=[0.2, 0.2]),
        Tf.Normalize(0.5, 0.5)
    ]),
    split="Training",
    sample_phase_annotations=True,
    map_phase_annotations=identity_
)


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


train_ds_2 = CATARACTS(
    root='/local/scratch/CATARACTS-to-Cataract101/',
    n_seq_frames=3,
    dt=1,
    transforms=Tf.Compose([
        Tf.Resize((128, 128)),
        Tf.RandomChoice([
                    Tf.RandomHorizontalFlip(),
                    Tf.RandomVerticalFlip(),
        ], p=[0.2, 0.2]),
        Tf.Normalize(0.5, 0.5)
    ]),
    phases=[3, 4, 5, 6, 7, 10, 13, 14],
    # phases=[0, 3, 4],
    split='Training',
    sample_phase_annotations=True,
    map_phase_annotations=map_phases_,
)


def map_(t: torch.Tensor):
    pass


train_ds = ConcatDataset([train_ds_1, train_ds_2])
val_ds = Cataract101(
    root='/local/scratch/cataract-101-processed/',
    n_seq_frames=T,
    dt=1,
    transforms=Tf.Compose([
        Tf.Resize((128, 128)),
        Tf.Normalize(0.5, 0.5)
    ]),
    split="Validation",
    sample_phase_annotations=True,
    map_phase_annotations=map_
)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, drop_last=True)

print("##### Loading model")
m = PhaseClassifier(n_seq_frames=3, n_classes=11, dim_multiplier=2).to(dev)
optim = torch.optim.Adam(m.parameters(), lr=0.0002, weight_decay=0.001, betas=(0.95, 0.5))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, factor=0.9, patience=5)
cutout = CoarseDropout(max_width=4, max_height=4)

print("##### training phase classification model:")
time.sleep(0.1)
best_val = 0
train_losses = []
val_losses = []
val_accs = []
lrs = []
for epoch in range(EPOCHS):

    loss_per_sample = []
    iter_dl = iter(train_dl)
    for i in tqdm(range(STEPS)):
        with torch.no_grad():
            sample = next(iter_dl)

        img_seq = sample['img_seq']
        phase_seq = sample['phase_seq']
        N, T, C, H, W = img_seq.shape

        optim.zero_grad()

        input = img_seq.view((N, T * C, H, W)).to(dev)
        target = phase_seq[:, -1].to(dev)
        prediction = m(input)

        loss = F.cross_entropy(prediction, target)
        loss_per_sample.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optim.step()

    avg_train_loss = np.mean(loss_per_sample)
    time.sleep(0.1)
    print(f"Epoch {epoch} Avg. Loss {avg_train_loss} LR {get_lr(optim)}")
    train_losses.append(avg_train_loss)
    lrs.append(get_lr(optim))
    time.sleep(0.1)

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
                target = sample['phase_seq'][:, -1].to(dev)
                N, T, C, H, W = img_seq.shape
                input = img_seq.view((N, T * C, H, W)).to(dev)
                targets = target if targets is None \
                    else torch.cat([targets, target], dim=0)
                prediction = m(input)
                predictions = prediction if predictions is None \
                    else torch.cat([predictions, prediction], dim=0)
                loss = F.cross_entropy(prediction, target)
                loss_per_sample.append(loss.item())

            avg_val_loss = np.mean(loss_per_sample)
            correct_predictions = (predictions.argmax(dim=-1) == targets).float()
            acc = (correct_predictions.sum() / len(correct_predictions)).item()
            time.sleep(0.1)
            print(f"Avg. val. loss {avg_val_loss} Acc {acc}")
            val_losses.append(avg_val_loss)
            val_accs.append(acc)
            time.sleep(0.1)

            if acc > best_val:
                best_val = acc
                torch.save(m.state_dict(), DIR + 'phase_model_extended.pth')
                print("Checkpoint saved.")

        m.train(True)

    # End of epoch
    scheduler.step(avg_train_loss)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].grid(True, axis='y')
    ax[0].plot(np.linspace(0, epoch, len(train_losses)), train_losses, color='blue', label='train-ce')
    ax[0].plot(np.linspace(0, epoch, len(val_losses)), val_losses, color='red', label='val-ce')
    ax[0].plot(np.linspace(0, epoch, len(val_accs)), val_accs, color='orange', label='val-acc')
    ax[0].legend()
    ax[1].plot(np.linspace(0, epoch, len(lrs)), lrs, color='green', label='lr')
    ax[1].legend()
    plt.autoscale()
    plt.savefig(DIR + '/phase_model_extended.png')
    plt.close()
