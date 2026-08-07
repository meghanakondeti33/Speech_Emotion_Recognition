"""
Prepare Dataset for Training

This script preprocesses every audio file and saves the extracted
features as NumPy arrays.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing import (
    preprocess_audio,
    extract_mel_spectrogram,
    extract_mfcc
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "metadata.csv"

MEL_FOLDER = PROJECT_ROOT / "data" / "processed" / "mel"

MFCC_FOLDER = PROJECT_ROOT / "data" / "processed" / "mfcc"

OUTPUT_METADATA = PROJECT_ROOT / "data" / "processed" / "processed_metadata.csv"


MEL_FOLDER.mkdir(parents=True, exist_ok=True)
MFCC_FOLDER.mkdir(parents=True, exist_ok=True)


def prepare_dataset():

    metadata = pd.read_csv(METADATA_PATH)

    processed_records = []

    print(f"Processing {len(metadata)} audio files...\n")

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing Audio"):

        actor_folder = f"Actor_{int(row['actor']):02d}"

        filepath = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "RAVDESS"
            / actor_folder
            / row["filename"]
        )

        audio = preprocess_audio(filepath)

        mel = extract_mel_spectrogram(audio)

        mfcc = extract_mfcc(audio)

        mel_name = f"sample_{idx:04d}.npy"

        mfcc_name = f"sample_{idx:04d}.npy"

        mel_path = MEL_FOLDER / mel_name

        mfcc_path = MFCC_FOLDER / mfcc_name

        np.save(mel_path, mel)

        np.save(mfcc_path, mfcc)

        processed_records.append({

            "mel_path": str(mel_path),

            "mfcc_path": str(mfcc_path),

            "emotion": row["emotion"],

            "actor": row["actor"],

            "gender": row["gender"]

        })

    processed_df = pd.DataFrame(processed_records)

    processed_df.to_csv(

        OUTPUT_METADATA,

        index=False

    )

    print("\nDataset preparation completed!")

    print(f"Metadata saved to: {OUTPUT_METADATA}")


if __name__ == "__main__":

    prepare_dataset()