"""
visualization.py — Utilities for saving generated-image grids.

These functions are used during training to periodically snapshot
what the Generator is producing.  A *fixed* noise batch is used so
that the same latent vectors are decoded at every checkpoint,
making visual progress easy to compare across epochs.
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.utils as vutils


def save_image_grid(
    images: torch.Tensor,
    filepath: str,
    nrow: int = 8,
    normalize: bool = True,
):
    """
    Save a batch of images as a single grid PNG.

    Parameters
    ----------
    images   : Tensor of shape (N, C, H, W) with values in [-1, 1].
    filepath : Where to write the PNG file.
    nrow     : Number of images per row in the grid.
    normalize: If True, rescale from [-1, 1] to [0, 1] for display.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # torchvision's save_image handles normalisation and grid layout
    vutils.save_image(
        images,
        filepath,
        nrow=nrow,
        normalize=normalize,
        value_range=(-1, 1),
        padding=2,
    )


def generate_and_save_samples(
    generator: torch.nn.Module,
    fixed_noise: torch.Tensor,
    epoch: int,
    sample_dir: str,
):
    """
    Run the generator on a fixed noise batch and save the result.

    Parameters
    ----------
    generator   : The Generator network (in eval mode is fine).
    fixed_noise : Tensor of shape (N, latent_dim), kept constant
                  across all epochs for fair visual comparison.
    epoch       : Current epoch number (used in the filename).
    sample_dir  : Directory to save the image grid.

    Returns
    -------
    filepath : str — Path to the saved image.
    """
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise)
    generator.train()

    filename = f"epoch_{epoch:04d}.png"
    filepath = os.path.join(sample_dir, filename)
    save_image_grid(fake_images, filepath)
    return filepath
