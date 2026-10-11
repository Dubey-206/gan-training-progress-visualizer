"""
Smoke test — verify that Generator & Discriminator forward passes
produce tensors of the expected shape, and that a single training
step runs without errors.
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import config
from src.models import Generator, Discriminator

print("=" * 50)
print("  SMOKE TEST")
print("=" * 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 1. Generator forward pass ────────────────────────────
G = Generator(latent_dim=config.LATENT_DIM, img_shape=config.IMG_SHAPE).to(device)
z = torch.randn(4, config.LATENT_DIM, device=device)
fake_imgs = G(z)
assert fake_imgs.shape == (4, 1, 28, 28), f"Bad G output shape: {fake_imgs.shape}"
print(f"[PASS] Generator output shape: {fake_imgs.shape}")

# ── 2. Discriminator forward pass ────────────────────────
D = Discriminator(img_shape=config.IMG_SHAPE).to(device)
validity = D(fake_imgs)
assert validity.shape == (4, 1), f"Bad D output shape: {validity.shape}"
assert (validity >= 0).all() and (validity <= 1).all(), "D output out of [0,1]"
print(f"[PASS] Discriminator output shape: {validity.shape}, range OK")

# ── 3. Loss + backward pass ─────────────────────────────
criterion = torch.nn.BCELoss()
real_labels = torch.ones(4, 1, device=device)
loss = criterion(validity, real_labels)
loss.backward()
print(f"[PASS] Backward pass completed, loss = {loss.item():.4f}")

# ── 4. Parameter counts ─────────────────────────────────
g_params = sum(p.numel() for p in G.parameters())
d_params = sum(p.numel() for p in D.parameters())
print(f"[INFO] Generator parameters:     {g_params:,}")
print(f"[INFO] Discriminator parameters:  {d_params:,}")

# ── 5. Dataset loading (download only) ──────────────────
print("\n[INFO] Testing MNIST download ...")
from src.dataset import get_mnist_dataloader
dataloader = get_mnist_dataloader(
    data_dir=config.DATA_DIR,
    batch_size=config.BATCH_SIZE,
)
batch_imgs, batch_labels = next(iter(dataloader))
assert batch_imgs.shape == (config.BATCH_SIZE, 1, 28, 28)
print(f"[PASS] DataLoader batch shape: {batch_imgs.shape}")
print(f"[PASS] Pixel range: [{batch_imgs.min():.2f}, {batch_imgs.max():.2f}]")

print("\n" + "=" * 50)
print("  ALL SMOKE TESTS PASSED [OK]")
print("=" * 50)
