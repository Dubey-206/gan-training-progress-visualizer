"""
config.py — Central configuration for GAN training.

All hyperparameters and paths are defined here so that they can be
tuned from a single location.  The values below are sensible defaults
for training a simple GAN on MNIST with a laptop CPU or a single GPU.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
# Root directory of the project (the folder that contains this file)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# ──────────────────────────────────────────────
# Model architecture
# ──────────────────────────────────────────────
LATENT_DIM = 100          # Dimension of the noise vector z
IMG_CHANNELS = 1          # MNIST is grayscale (1 channel)
IMG_SIZE = 28             # MNIST images are 28×28 pixels
IMG_SHAPE = (IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

# Generator hidden-layer width
G_HIDDEN = 256

# Discriminator hidden-layer width
D_HIDDEN = 256

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
BATCH_SIZE = 64
NUM_EPOCHS = 20           # Total training epochs
LEARNING_RATE = 0.0002
BETAS = (0.5, 0.999)     # Adam optimizer momentum terms

# ──────────────────────────────────────────────
# Sampling & logging
# ──────────────────────────────────────────────
# How many epochs between saving a sample-image grid
SAMPLE_INTERVAL = 2

# Number of images in the fixed-noise batch used for visual comparison
NUM_FIXED_SAMPLES = 64

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
SEED = 42
