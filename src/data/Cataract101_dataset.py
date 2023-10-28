import os

import numpy as np
import pandas as pd
import torch

from natsort import natsorted

from .eye_surgery_dataset import EyeSurgeryDataset, SamplePt


class Cataract101(EyeSurgeryDataset):

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

        super(Cataract101, self).__init__(root, n_seq_frames, dt, split, phases, sample_imgs, sample_phase_annotations,
                                          map_phase_annotations, transforms)

        self.file_end = file_end
        self.phase_annotation_pd = pd.read_csv(root + 'annotations.csv', sep=';')
        self.max_phase = self.phase_annotation_pd['Phase'].max()
        self.samples = self.read_splits(phases)

    def map_video_path_to_case_frame(self, path: str) -> (str, int):
        """ Returns case/video id and frame number from given video path."""
        splits = path.split("/")
        frame_nr = int(splits[-1].replace("frame", "").replace(self.file_end, ""))
        case_id = splits[-2]
        return case_id, frame_nr

    def map_case_frame_to_phase(self, case_id: str, frame_nr: int) -> torch.Tensor:
        """ Returns one-hot tensor for phase annotation for given case id and frame number."""
        case_nr = int(case_id.split('_')[-1])
        phase_annotation = torch.zeros((self.max_phase + 1,))
        case_pd = self.phase_annotation_pd[self.phase_annotation_pd['VideoID'] == case_nr]
        _idx = np.where([case_pd['FrameNo'].values < frame_nr])[1]
        if len(_idx > 0):
            phase_annotation[case_pd['Phase'].values[_idx[-1]]] += 1
        else:
            phase_annotation[0] += 1
        return phase_annotation

    def filter_frame_paths_by_phases(self, list_of_paths: list, list_of_valid_phases: list) -> list:
        new = []
        for path in list_of_paths:
            case_id, frame_nr = self.map_video_path_to_case_frame(path)
            phase_tensor = self.map_case_frame_to_phase(case_id, frame_nr)
            phase = torch.argmax(phase_tensor)
            if phase in list_of_valid_phases:
                new.append(path)
        return new

    def read_splits(self, phases: list = None) -> list:
        """ Determine every possible video/case in the split.

        :return: List of SamplePt objects
        """

        if self.split == 'Training':
            start = 0
            stop = int(len(natsorted(os.listdir(self.root)))*0.7)
        elif self.split == 'Validation':
            start = int(len(natsorted(os.listdir(self.root)))*0.7) + 1
            stop = int(len(natsorted(os.listdir(self.root)))*0.9)
        elif self.split == 'Test':
            start = int(len(natsorted(os.listdir(self.root))) * 0.9) + 1
            stop = int(len(natsorted(os.listdir(self.root))))
        else:
            start = 0
            stop = len(os.listdir(self.root))

        samples = []
        c = 0
        for video in natsorted(os.listdir(self.root))[start:stop]:

            video_path = self.root + video + "/"
            if not os.path.isdir(video_path):
                continue
            list_of_frame_paths = [video_path + img for img in natsorted(os.listdir(video_path))]
            # Filter according to list of wanted phases
            if phases is not None:
                list_of_frame_paths = self.filter_frame_paths_by_phases(list_of_frame_paths, phases)
            if self.n_seq_frames*self.dt > len(list_of_frame_paths):
                raise ValueError("Combination of #frames and dt exceeds max. number of frames.")
            if self.n_seq_frames == -1:
                sample = SamplePt(seq_id=video, sample_id=c)
                sample.set_frame_paths(list_of_frame_paths)
                samples.append(sample)
                c += 1
            else:
                for frame_id in np.arange(0, len(list_of_frame_paths), self.dt):
                    if frame_id + self.n_seq_frames < len(list_of_frame_paths):
                        sample = SamplePt(seq_id=video, sample_id=c)
                        sample.set_frame_paths(list_of_frame_paths[frame_id:frame_id + self.n_seq_frames])
                        samples.append(sample)
                        c += 1
        return samples
