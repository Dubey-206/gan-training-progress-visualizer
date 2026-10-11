"""
dataset.py — MNIST dataset loading and preprocessing.

The MNIST dataset contains 60 000 training images of handwritten digits
(0–9), each 28×28 pixels in grayscale.

Preprocessing
-------------
1. Convert to Tensor           → pixel values go from [0, 255] to [0, 1].
2. Normalize to mean=0.5, std=0.5  → pixel values become [-1, 1].
   This matches the Tanh output range of the Generator.
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloader(data_dir: str, batch_size: int, num_workers: int = 0):
    """
    Download MNIST (if necessary) and return a DataLoader.

    Parameters
    ----------
    data_dir    : Path where the dataset will be stored / cached.
    batch_size  : Number of images per mini-batch.
    num_workers : Parallel data-loading workers (0 = main thread only,
                  safest default for Windows).

    Returns
    -------
    DataLoader yielding (images, labels) batches.
    Images are Tensors of shape (batch_size, 1, 28, 28) with values in [-1, 1].
    """
    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Compose the transform pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),                          # [0, 255] → [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5]),    # [0, 1]   → [-1, 1]
    ])

    # Download MNIST if it isn't already present
    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,   # drop the last incomplete batch for stable training
    )

    return dataloader
