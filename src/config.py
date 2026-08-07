"""
Project Configuration File
Speech Emotion Recognition
"""

# ==============================
# Audio Configuration
# ==============================

SAMPLE_RATE = 16000
DURATION = 3
TARGET_LENGTH = SAMPLE_RATE * DURATION

# ==============================
# Feature Extraction
# ==============================

N_MELS = 128
N_MFCC = 40

# ==============================
# Dataset
# ==============================

NUM_CLASSES = 8

EMOTIONS = [
    "Neutral",
    "Calm",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprised"
]

# ==============================
# Training
# ==============================

BATCH_SIZE = 32

LEARNING_RATE = 1e-3

EPOCHS = 30

RANDOM_STATE = 42

# ==============================
# Model
# ==============================

DROPOUT = 0.5