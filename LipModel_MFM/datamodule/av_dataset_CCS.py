import csv
import os

import torch
import torchaudio
import torchvision

from .hand_util import load_hand_recog


def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
    return waveform.transpose(1, 0)


class AVDataset_CCS(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir,
            label_path,
            subset,
            modality,
            audio_transform,
            video_transform,
            rate_ratio=640,
    ):

        self.root_dir = root_dir

        self.modality = modality
        self.rate_ratio = rate_ratio

        self.list = self.load_list(label_path)

        self.audio_transform = audio_transform
        self.video_transform = video_transform

    def load_list(self, label_path):
        paths_counts_labels = []
        with open(label_path, newline="", encoding="utf-8-sig") as stream:
            for line_number, row in enumerate(csv.reader(stream), start=1):
                if not row or (row[0].lstrip().startswith("#")):
                    continue
                if len(row) != 6:
                    raise ValueError(
                        f"{label_path}:{line_number} must contain 6 fields, "
                        f"received {len(row)}."
                    )
                (
                    dataset_name,
                    rel_path,
                    input_length,
                    token_id,
                    hand_recog_path,
                    hand_position_path,
                ) = row
                paths_counts_labels.append(
                    (
                        dataset_name,
                        rel_path,
                        int(input_length),
                        torch.tensor([int(value) for value in token_id.split()]),
                        hand_recog_path,
                        hand_position_path,
                    )
                )
        return paths_counts_labels

    def _resolve_path(self, path):
        return path if os.path.isabs(path) else os.path.join(self.root_dir, path)

    def __getitem__(self, idx):
        dataset_name, rel_path, input_length, token_id, hand_recog_path, hand_position_path = self.list[idx]
        path = os.path.join(self.root_dir, dataset_name, rel_path)
        if os.path.exists(path) is False:
            raise FileNotFoundError(f"{path} does not exist.")
        hand_recog_path = self._resolve_path(hand_recog_path)
        hand_position_path = self._resolve_path(hand_position_path)
        if self.modality == "video":
            if not os.path.exists(hand_recog_path):
                raise FileNotFoundError(f"{hand_recog_path} does not exist.")
            if not os.path.exists(hand_position_path):
                raise FileNotFoundError(f"{hand_position_path} does not exist.")
            video = load_video(path)
            video = self.video_transform(video)
            if input_length != len(video):
                raise ValueError(
                    f"Label length {input_length} does not match decoded video "
                    f"length {len(video)} for {path}."
                )
            hand_recog_matrix = load_hand_recog(
                hand_recog_path, hand_position_path, len(video)
            )
            return {"input": video, "target": token_id, "hand_matrix": hand_recog_matrix}
        elif self.modality == "audio":
            audio = load_audio(path)
            audio = self.audio_transform(audio)
            return {"input": audio, "target": token_id}
        elif self.modality == "audiovisual":
            video = load_video(path)
            audio = load_audio(path)
            audio = cut_or_pad(audio, len(video) * self.rate_ratio)
            video = self.video_transform(video)
            audio = self.audio_transform(audio)
            return {"video": video, "audio": audio, "target": token_id}

    def __len__(self):
        return len(self.list)
