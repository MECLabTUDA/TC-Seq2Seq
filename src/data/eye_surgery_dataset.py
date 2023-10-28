import os
import time
import warnings

import torch
from torch.utils.data import Dataset
import torchvision.transforms as Tf
import PIL.Image as Image

TT = Tf.ToTensor()


class SamplePt:

    """ Abstract sample object. """

    def __init__(self,
                 seq_id: str,
                 sample_id: int):

        self.phase_annotations = None
        self.frame_paths = []  # List of paths to frames
        self.label_paths = []  # List of paths to labels
        self.frame_nrs = []  # Frame time-stamps of the sequence the frames are from
        self.seq_id = seq_id  # ID of the video/case the frame is from
        self.sample_id = sample_id  # Incrementer for sample number

    def set_frame_paths(self, list_of_paths):
        self.frame_paths = list_of_paths

    def set_label_paths(self, list_of_label_paths):
        self.label_paths = list_of_label_paths


class EyeSurgeryDataset(Dataset):

    def __init__(self,
                 root: str,
                 n_seq_frames: int = 1,
                 dt: int = 1,
                 split: str = 'Training',
                 phases: list = None,
                 sample_imgs: bool = True,
                 sample_phase_annotations: bool = False,
                 map_phase_annotations=None,
                 transforms=None):

        super(EyeSurgeryDataset, self).__init__()

        assert os.path.isdir(root), "Could not read root dir"
        self.root = root
        assert split in [None, 'Training', 'Validation', 'Test']
        self.split = split
        self.n_seq_frames = n_seq_frames
        self.dt = dt
        self.T = transforms
        self.sample_imgs = sample_imgs
        self.sample_phase_annotations = sample_phase_annotations
        self.map_phase_annotations = map_phase_annotations
        if self.map_phase_annotations is not None:
            warnings.warn("Passing a phase map converts the label structure"
                          " from (N, C) one-hot to (N,) integer values.")

        self.samples = None  # List of SamplePt objects

    def map_video_path_to_case_frame(self, path: str) -> (str, int):
        raise NotImplementedError

    def map_case_frame_to_phase(self, case_id: str, frame_nr: int) -> torch.Tensor:
        raise NotImplementedError

    def read_splits(self, phase_list: list) -> list:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> dict:

        sample = {}
        img_seq = None
        phase_annotation_seq = None
        frame_nrs_list = []
        case_id = None
        for img_path in self.samples[item].frame_paths:
            # Read meta-data
            _case_id, frame_nr = self.map_video_path_to_case_frame(img_path)
            frame_nrs_list.append(frame_nr)

            if case_id is None:
                case_id = _case_id
            elif case_id != _case_id:
                raise ValueError("Frames not from same case")

            if self.sample_phase_annotations:
                # Read annotations
                phase_annotation = self.map_case_frame_to_phase(case_id, frame_nr)
                if self.map_phase_annotations is not None:
                    phase_annotation = torch.argmax(phase_annotation, dim=-1)
                    self.map_phase_annotations(phase_annotation)

            else:
                phase_annotation = torch.empty([])

            phase_annotation_seq = phase_annotation.unsqueeze(0) if phase_annotation_seq is None \
                else torch.cat([phase_annotation_seq, phase_annotation.unsqueeze(0)], dim=0)

            if self.sample_imgs:
                # Read image
                pil_img = Image.open(img_path)
                torch_img = TT(pil_img)
                img_seq = torch_img.unsqueeze(0) if img_seq is None \
                    else torch.cat([img_seq, torch_img.unsqueeze(0)], dim=0)
            else:
                img_seq = torch.empty([])

        if self.sample_imgs:
            img_seq = self.T(img_seq) if self.T is not None else img_seq

        sample['img_seq'] = img_seq
        sample['label_seq'] = torch.empty([])
        sample['phase_seq'] = phase_annotation_seq
        sample['case_id'] = case_id
        sample['frame_nrs'] = frame_nrs_list
        return sample


class Img2ImgDataset(Dataset):

    def __init__(self,
                 ds_A: EyeSurgeryDataset,
                 ds_B: EyeSurgeryDataset):
        self.ds_A = ds_A
        self.ds_B = ds_B

    def __len__(self):
        return max(len(self.ds_A), len(self.ds_B))

    def __getitem__(self, item) -> dict:
        item_A = item if item < len(self.ds_A) else item % len(self.ds_A)
        item_B = item if item < len(self.ds_B) else item % len(self.ds_B)

        return {'A': self.ds_A[item_A]['img_seq'],
                'B': self.ds_B[item_B]['img_seq'],
                'paths_A': self.ds_A.samples[item_A].frame_paths,
                'paths_B': self.ds_B.samples[item_B].frame_paths,
                'id_A': self.ds_A[item_A]['case_id'],
                'id_B': self.ds_B[item_B]['case_id'],
                'frame_nrs_A': self.ds_A[item_A]['frame_nrs'],
                'frame_nrs_B': self.ds_B[item_B]['frame_nrs']
                }
