"""
Audio Preprocessing Module
Speech Emotion Recognition
"""

import librosa
import numpy as np

from config import (
    SAMPLE_RATE,
    TARGET_LENGTH,
    N_MELS,
    N_MFCC
)


# ==========================================================
# Load Audio
# ==========================================================

def load_audio(audio_path):
    """
    Load an audio file.

    Parameters
    ----------
    audio_path : str

    Returns
    -------
    np.ndarray
        Audio samples

    int
        Sampling rate
    """

    audio, sr = librosa.load(
        audio_path,
        sr=None
    )

    return audio, sr


# ==========================================================
# Trim Silence
# ==========================================================

def trim_audio(audio):

    trimmed_audio, _ = librosa.effects.trim(audio)

    return trimmed_audio


# ==========================================================
# Resample Audio
# ==========================================================

def resample_audio(audio, original_sr):

    if original_sr == SAMPLE_RATE:
        return audio

    return librosa.resample(
        audio,
        orig_sr=original_sr,
        target_sr=SAMPLE_RATE
    )


# ==========================================================
# Normalize Audio
# ==========================================================

def normalize_audio(audio):

    return librosa.util.normalize(audio)


# ==========================================================
# Pad / Truncate
# ==========================================================

def fix_length(audio):

    if len(audio) < TARGET_LENGTH:

        audio = np.pad(
            audio,
            (0, TARGET_LENGTH - len(audio))
        )

    else:

        audio = audio[:TARGET_LENGTH]

    return audio


# ==========================================================
# Mel Spectrogram
# ==========================================================

def extract_mel_spectrogram(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db


# ==========================================================
# MFCC
# ==========================================================

def extract_mfcc(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    return mfcc


# ==========================================================
# Complete Pipeline
# ==========================================================

def preprocess_audio(audio_path):

    audio, sr = load_audio(audio_path)

    audio = trim_audio(audio)

    audio = resample_audio(audio, sr)

    audio = normalize_audio(audio)

    audio = fix_length(audio)

    return audio