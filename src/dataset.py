"""
PyTorch Dataset
"""

import torch
import pandas as pd

from torch.utils.data import Dataset

from preprocessing import (
    preprocess_audio,
    extract_mel_spectrogram
)

emotion_to_label = {
    "Neutral":0,
    "Calm":1,
    "Happy":2,
    "Sad":3,
    "Angry":4,
    "Fearful":5,
    "Disgust":6,
    "Surprised":7
}


class EmotionDataset(Dataset):

    def __init__(self, metadata_path):

        self.metadata = pd.read_csv(metadata_path)

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self,index):

        sample = self.metadata.iloc[index]

        audio = preprocess_audio(
            sample["filepath"]
        )

        mel = extract_mel_spectrogram(audio)

        mel = torch.tensor(
            mel,
            dtype=torch.float32
        )

        mel = mel.unsqueeze(0)

        label = emotion_to_label[
            sample["emotion"]
        ]

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return mel,label