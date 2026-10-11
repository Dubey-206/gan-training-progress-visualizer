"""
models.py — Generator and Discriminator networks for a vanilla GAN.

Both networks are simple fully-connected (MLP) architectures.  This is
intentional: an MLP-based GAN is easier to understand, debug, and
explain in a viva than a convolutional architecture, and it is more
than sufficient for 28×28 grayscale MNIST digits.

Key concepts
------------
Generator   : Takes a random noise vector z ∈ ℝ^LATENT_DIM and maps it
              to a fake image of shape (1, 28, 28).
Discriminator: Takes an image of shape (1, 28, 28) and outputs a single
              probability — how likely the image is *real* (vs. fake).
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Maps a latent noise vector z → fake image.

    Architecture
    ------------
    z (100) → 256 → 512 → 1024 → 784  →  reshape to (1, 28, 28)

    Each hidden layer uses:
      • Linear transformation
      • Batch Normalization (stabilises training)
      • LeakyReLU activation
    The output layer uses Tanh to scale pixel values to [-1, 1].
    """

    def __init__(self, latent_dim: int, img_shape: tuple):
        super().__init__()
        self.img_shape = img_shape  # e.g. (1, 28, 28)

        # Total number of pixels in the output image
        img_flat = 1
        for s in img_shape:
            img_flat *= s

        def block(in_feat, out_feat, normalize=True):
            """Helper: one dense layer + optional batch-norm + activation."""
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, img_flat),
            nn.Tanh(),  # output pixels in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : Tensor of shape (batch_size, latent_dim)

        Returns
        -------
        Tensor of shape (batch_size, *img_shape)
        """
        img_flat = self.model(z)
        img = img_flat.view(img_flat.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    """
    Maps an image → probability of being real.

    Architecture
    ------------
    (1, 28, 28) → flatten → 512 → 256 → 1  (sigmoid)

    Each hidden layer uses:
      • Linear transformation
      • LeakyReLU activation
      • Dropout (prevents the discriminator from overpowering the generator)
    """

    def __init__(self, img_shape: tuple):
        super().__init__()

        img_flat = 1
        for s in img_shape:
            img_flat *= s

        self.model = nn.Sequential(
            nn.Linear(img_flat, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid(),  # output ∈ (0, 1): probability of "real"
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        img : Tensor of shape (batch_size, *img_shape)

        Returns
        -------
        Tensor of shape (batch_size, 1) — probability the image is real.
        """
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
