import torch
from easydict import EasyDict
import torchvision.transforms as Tf
from torch.utils.data import DataLoader, Dataset, Subset

from src.data import CATARACTS, Cataract101, CaDISv2, Img2ImgDataset


def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    """ Maps back a normalized image in [-1, 1] into [0, 1]"""
    assert img_tensor.min() >= -1.0
    assert img_tensor.max() <= 1.0
    img_tensor = (img_tensor + 1.0) / 2.0
    assert img_tensor.min() >= 0.0
    assert img_tensor.max() <= 1.0
    return img_tensor


def get_ds_from_path(path: str, conf: dict) -> (Dataset, Dataset):
    """ Returns the correct datasets based on the given root path.

    """
    if "CATARACTS" in path:
        train_ds = CATARACTS(
            root=path,
            n_seq_frames=conf.data.seq_frames_train,
            dt=conf.data.dt,
            transforms=Tf.Compose([
                Tf.RandomCrop((1080, 1080)),
                Tf.Resize((conf.data.img_dim, conf.data.img_dim)),
                Tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            phases=[0, 3, 4, 5, 6, 7, 8, 10, 13, 14],
            split="Training")
        test_ds = CATARACTS(
            root=path,
            n_seq_frames=conf.data.seq_frames_test,
            dt=conf.data.dt,
            transforms=Tf.Compose([
                Tf.Resize((conf.data.img_dim, conf.data.img_dim)),
                Tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            phases=[0, 3, 4, 5, 6, 7, 8, 10, 13, 14],
            split="Test")
    elif "cataract-101" in path:
        train_ds = Cataract101(
            root=path,
            n_seq_frames=conf.data.seq_frames_train,
            dt=conf.data.dt,
            transforms=Tf.Compose([
                Tf.RandomCrop((540, 540)),
                Tf.Resize((conf.data.img_dim, conf.data.img_dim)),
                Tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            phases=[0, 1, 2, 3, 4, 5, 6, 8],
            split="Training")
        test_ds = Cataract101(
            root=path,
            n_seq_frames=conf.data.seq_frames_test,
            dt=conf.data.dt,
            transforms=Tf.Compose([
                Tf.Resize((conf.data.img_dim, conf.data.img_dim)),
                Tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            phases=[0, 1, 2, 3, 4, 5, 6, 8],
            split="Test")
    else:
        raise ValueError("Could not read dataset type from root path.")
    return train_ds, test_ds


def get_dataloaders(conf: dict, shuffle_test: bool = True, subset_indices: list = None):
    train_ds_A, test_ds_A = get_ds_from_path(path=conf.data.root_A, conf=conf)
    train_ds_B, test_ds_B = get_ds_from_path(path=conf.data.root_B, conf=conf)
    print(f"Domain A --- Training: {len(train_ds_A)} --- Testing: {len(test_ds_A)}")
    print(f"Domain B --- Training: {len(train_ds_B)} --- Testing: {len(test_ds_B)}")
    train_ds = Img2ImgDataset(ds_A=train_ds_A, ds_B=train_ds_B)
    test_ds = Img2ImgDataset(ds_A=test_ds_A, ds_B=test_ds_B)
    if subset_indices is not None:
        assert len(subset_indices) > 0
        train_ds = Subset(train_ds, subset_indices)
        test_ds = Subset(test_ds, subset_indices)
    train_dl = DataLoader(train_ds, batch_size=conf.training.batch_size, num_workers=conf.data.num_workers,
                          shuffle=True, drop_last=True, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=conf.testing.batch_size, num_workers=conf.data.num_workers,
                         shuffle=shuffle_test, drop_last=True, pin_memory=True)
    return train_dl, test_dl


def edict_to_dict(dic: EasyDict) -> dict:
    return dic.__dict__


class UNIT_Transform(object):

    def __init__(self, gen_source: torch.nn.Module, gen_target: torch.nn.Module):
        self.gen_source = gen_source
        self.gen_target = gen_target

    def __call__(self, sample_img):
        with torch.no_grad():
            h_source, n_source = self.gen_source.encode(sample_img)
            translated_img = self.gen_target.decode(h_source + n_source)
        return translated_img
