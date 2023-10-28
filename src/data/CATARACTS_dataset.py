import os

import numpy as np
import pandas as pd
import torch
from natsort import natsorted

from .eye_surgery_dataset import EyeSurgeryDataset, SamplePt


class CATARACTS(EyeSurgeryDataset):

    def __init__(self,
                 root: str,
                 n_seq_frames: int = 1,
                 dt: int = 1,
                 split: str = 'Training',
                 phases: list = None,
                 transforms=None,
                 sample_imgs: bool = True,
                 sample_phase_annotations: bool = False,
                 map_phase_annotations=None,
                 file_end: str = '.jpg'):
        super(CATARACTS, self).__init__(root, n_seq_frames, dt, split, phases, sample_imgs, sample_phase_annotations,
                                        map_phase_annotations, transforms)

        self.file_end = file_end
        self.phase_annotations = self.read_annotations()
        self.samples = self.read_splits(phases)

    def read_annotations(self) -> dict:
        annotation_dict = {}
        for case in natsorted(os.listdir(self.root + "phase_annotations/")):
            if not ((case.startswith("train") and self.split == "Training")
                    or (case.startswith("test") and self.split == "Test")):
                continue
            annotation_path = self.root + "phase_annotations/" + case
            case = case.split(".")[0]
            annotation_dict[case] = pd.read_csv(annotation_path)
        return annotation_dict

    def map_case_frame_to_phase(self, case: str, frame_nr: int) -> torch.Tensor:
        phase_tensor = torch.zeros(size=(19,))
        pd_frame = self.phase_annotations[case]
        phase = pd_frame[pd_frame['Frame'] == (frame_nr + 1)]['Steps'].values[0]
        phase_tensor[phase] += 1
        return phase_tensor

    def map_video_path_to_case_frame(self, path: str) -> (str, int):
        path = path.split("/")
        case_id = path[-2]
        frame_nr = int(path[-1].replace(self.file_end, "").replace("frame", ""))
        return case_id, frame_nr

    def filter_frame_paths_by_phases(self, list_of_paths: list, list_of_valid_phases: list) -> list:
        new = []
        for path in list_of_paths:
            case_id, frame_nr = self.map_video_path_to_case_frame(path)
            phase_tensor = self.map_case_frame_to_phase(case_id, frame_nr)
            phase = torch.argmax(phase_tensor)
            if phase in list_of_valid_phases:
                new.append(path)
        return new

    def read_splits(self, phases: list) -> list:
        samples = []
        c = 0
        for case in natsorted(os.listdir(self.root)):
            if not ((case.startswith("train") and self.split == "Training")
                    or (case.startswith("test") and self.split == "Test")):
                continue
            elif case.startswith("phase_annotations"):
                continue
            else:
                case_path = self.root + case + "/"
                list_of_frame_paths = [case_path + frame for frame in natsorted(os.listdir(case_path))]
                # Filter according to list of wanted phases
                if phases is not None:
                    list_of_frame_paths = self.filter_frame_paths_by_phases(list_of_frame_paths, phases)
                if self.n_seq_frames * self.dt > len(list_of_frame_paths):
                    raise ValueError("Combination of #frames and dt exceeds max. number of frames.")
                if self.n_seq_frames == -1:
                    sample = SamplePt(seq_id=case, sample_id=c)
                    sample.set_frame_paths(list_of_frame_paths)
                    samples.append(sample)
                    c += 1
                else:
                    for frame_id in np.arange(0, len(list_of_frame_paths), self.dt):
                        if frame_id + self.n_seq_frames < len(list_of_frame_paths):
                            sample = SamplePt(seq_id=case, sample_id=c)
                            sample.set_frame_paths(list_of_frame_paths[frame_id:frame_id + self.n_seq_frames])
                            samples.append(sample)
                            c += 1
        return samples
