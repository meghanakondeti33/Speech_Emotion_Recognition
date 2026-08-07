"""
PyTorch DataLoader
"""

import torch

from torch.utils.data import (
    DataLoader,
    random_split
)

from dataset import EmotionDataset

from config import (
    BATCH_SIZE,
    RANDOM_STATE
)


def create_dataset(metadata_path):

    return EmotionDataset(metadata_path)


def split_dataset(dataset):

    train_size = int(0.7 * len(dataset))

    val_size = int(0.15 * len(dataset))

    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_STATE)

    train_dataset, val_dataset, test_dataset = random_split(

        dataset,

        [train_size, val_size, test_size],

        generator=generator

    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(

    train_dataset,

    val_dataset,

    test_dataset

):

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False

    )

    return train_loader, val_loader, test_loader