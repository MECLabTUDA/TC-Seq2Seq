import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseClassifier(nn.Module):

    def __init__(self,
                 n_seq_frames: int = 1,
                 n_classes: int = 1,
                 dim_multiplier: int = 1):
        super(PhaseClassifier, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3 * n_seq_frames, 64*dim_multiplier, 4, stride=2, padding=1),
            nn.BatchNorm2d(64 * dim_multiplier),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64*dim_multiplier, 128*dim_multiplier, 4, stride=2, padding=1),
            nn.BatchNorm2d(128*dim_multiplier),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128*dim_multiplier, 256*dim_multiplier, 4, stride=2, padding=1),
            nn.BatchNorm2d(256*dim_multiplier),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256*dim_multiplier, 512*dim_multiplier, 4, padding=1),
            nn.BatchNorm2d(512*dim_multiplier),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512*dim_multiplier, n_classes, 4, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x
