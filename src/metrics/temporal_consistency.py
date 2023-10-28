import torch
import torch.nn.functional as F
import torchvision.transforms as Tf
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt

ToGray = Tf.Grayscale()


def residual_ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    #a = ToGray(a)
    #b = ToGray(b)
    (score, diff) = structural_similarity(a.cpu().numpy(), b.cpu().numpy(), full=True, channel_axis=0)
    return score


def residual_rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = F.mse_loss(a, b)
    return diff


def temporal_consistency_metric(real_seq: torch.Tensor,
                                translated_seq: torch.Tensor,
                                nt: int = -1,
                                dist_func: str = "RMSE") -> (float, torch.Tensor):
    """ Returns the temporal consistency metric and the distance matrix, comparing the distances
        of the residual maps between each time-step in the sequences.
    """
    assert real_seq.shape == translated_seq.shape
    N, T, C, H, W = real_seq.shape
    assert N == 1
    if dist_func == 'SSIM':
        d = residual_ssim
    elif dist_func == 'RMSE':
        d = residual_rmse
    else:
        raise ValueError("Unknown frame distance function for TC metric")
    distm = torch.zeros(size=(T, T))
    for t in range(T):
        for _t in range(T):
            if nt > 0 and (abs(_t - t) > nt):
                continue
            # Compute residual maps
            real_seq_rmap = torch.abs(real_seq[:, t] - real_seq[:, _t]).view((N*C, H, W))
            translated_seq_rmap = torch.abs(translated_seq[:, t] - translated_seq[:, _t]).view((N*C, H, W))
            distm[t, _t] = torch.FloatTensor([d(real_seq_rmap, translated_seq_rmap)])

    return torch.sum(distm).item()/T, distm

def time_seg_cluster_metric(prediction: torch.Tensor, target: torch.Tensor):
    prediction_cluster_count = 1
    target_cluster_count = 1
    correct_predictions = 0
    current_predicted_phase = prediction[0]
    current_target_phase = target[0]
    for t in range(1, prediction.shape[0]):
        if prediction[t] != current_predicted_phase:
            current_predicted_phase = prediction[t]
            prediction_cluster_count += 1
        if target[t] != current_target_phase:
            current_target_phase = target[t]
            target_cluster_count += 1
        if prediction[t] == target[t]:
            correct_predictions += 1
    return (correct_predictions/prediction.shape[0])*(1/(1 + abs(target_cluster_count - prediction_cluster_count)))
