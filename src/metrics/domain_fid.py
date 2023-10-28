import torch
import torchvision.transforms as tf
import numpy as np
from torchvision.models import Inception_V3_Weights
from scipy.linalg import sqrtm


def get_inception() -> torch.nn.Module:
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights=Inception_V3_Weights.DEFAULT)
    model.eval()
    return model


def calculate_fid(m: torch.nn.Module, images_X: torch.Tensor, images_YX: torch.Tensor) -> float:
    f_X = m(images_X).numpy()
    f_YX = m(images_YX).numpy()
    mu_X, sigma_X = f_X.mean(axis=0), np.cov(f_X, rowvar=False)
    mu_YX, sigma_YX = f_YX.mean(axis=0), np.cov(f_YX, rowvar=False)
    ssdiff = np.sum((mu_X - mu_YX)**2.0)
    covmean = sqrtm(sigma_X.dot(sigma_YX))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return ssdiff + np.trace(sigma_X + sigma_YX - 2.0 * covmean)



