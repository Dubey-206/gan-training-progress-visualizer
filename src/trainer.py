"""
trainer.py — GAN training loop with loss logging, checkpointing and
             periodic sample generation.

Training overview
-----------------
Each epoch iterates over every mini-batch of real MNIST images and
performs two update steps:

1. **Discriminator step**
   • Feed real images    → D should output ≈ 1  (real label)
   • Feed fake images    → D should output ≈ 0  (fake label)
   • Minimise Binary Cross-Entropy between D's predictions and labels.

2. **Generator step**
   • Generate fake images from noise.
   • Feed them to D      → G *wants* D to output ≈ 1
   • Minimise BCE so that D is fooled.

The adversarial interplay drives G to produce increasingly realistic
images and D to become a better detector — until (ideally) G produces
images indistinguishable from real data.
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from src.models import Generator, Discriminator
from src.dataset import get_mnist_dataloader
from src.visualization import generate_and_save_samples


class GANTrainer:
    """Encapsulates the full training procedure for a vanilla GAN."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : module or object whose attributes match those in config.py.
        """
        self.config = config

        # ── Device selection ──────────────────────────────────
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("[INFO] CUDA not available — training on CPU.")

        # ── Reproducibility ───────────────────────────────────
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)

        # ── Ensure output directories exist ───────────────────
        for d in [config.SAMPLE_DIR, config.METRICS_DIR, config.CHECKPOINT_DIR]:
            os.makedirs(d, exist_ok=True)

        # ── Data ──────────────────────────────────────────────
        self.dataloader = get_mnist_dataloader(
            data_dir=config.DATA_DIR,
            batch_size=config.BATCH_SIZE,
        )

        # ── Models ────────────────────────────────────────────
        self.generator = Generator(
            latent_dim=config.LATENT_DIM,
            img_shape=config.IMG_SHAPE,
        ).to(self.device)

        self.discriminator = Discriminator(
            img_shape=config.IMG_SHAPE,
        ).to(self.device)

        # ── Loss & optimizers ────────────────────────────────
        self.criterion = nn.BCELoss()

        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.LEARNING_RATE,
            betas=config.BETAS,
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.LEARNING_RATE,
            betas=config.BETAS,
        )

        # ── Fixed noise for consistent sample generation ─────
        self.fixed_noise = torch.randn(
            config.NUM_FIXED_SAMPLES, config.LATENT_DIM, device=self.device
        )

        # ── Metrics tracking ─────────────────────────────────
        self.metrics_history = []  # list of dicts, one per epoch
        self.start_epoch = 0      # overwritten if resuming from checkpoint

    # ──────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────
    def save_checkpoint(self, epoch: int):
        """Save model weights, optimizer states and epoch counter."""
        checkpoint = {
            "epoch": epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
            "fixed_noise": self.fixed_noise,
            "metrics_history": self.metrics_history,
        }
        path = os.path.join(self.config.CHECKPOINT_DIR, "latest_checkpoint.pt")
        torch.save(checkpoint, path)
        print(f"[CHECKPOINT] Saved -> {path}")

    def load_checkpoint(self):
        """
        Resume training from the latest checkpoint, if one exists.

        Returns True if a checkpoint was loaded, False otherwise.
        """
        path = os.path.join(self.config.CHECKPOINT_DIR, "latest_checkpoint.pt")
        if not os.path.isfile(path):
            print("[INFO] No checkpoint found — starting from scratch.")
            return False

        print(f"[INFO] Loading checkpoint from {path} ...")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        self.optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        self.fixed_noise = checkpoint["fixed_noise"]
        self.metrics_history = checkpoint.get("metrics_history", [])
        self.start_epoch = checkpoint["epoch"] + 1

        print(f"[INFO] Resumed from epoch {checkpoint['epoch']}. "
              f"Will continue from epoch {self.start_epoch}.")
        return True

    # ──────────────────────────────────────────────────────────
    # Metrics I/O
    # ──────────────────────────────────────────────────────────
    def save_metrics(self):
        """Write the accumulated metrics to a CSV file."""
        if not self.metrics_history:
            return
        df = pd.DataFrame(self.metrics_history)
        csv_path = os.path.join(self.config.METRICS_DIR, "training_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"[METRICS] Saved -> {csv_path}")

    # ──────────────────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────────────────
    def train(self, resume: bool = True):
        """
        Run the full training loop.

        Parameters
        ----------
        resume : If True, attempt to load a saved checkpoint first.
        """
        if resume:
            self.load_checkpoint()

        cfg = self.config
        total_epochs = cfg.NUM_EPOCHS

        print(f"\n{'='*60}")
        print(f"  GAN Training - {total_epochs} epochs, batch size {cfg.BATCH_SIZE}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        # ── Save initial samples (epoch 0, before any training) ──
        if self.start_epoch == 0:
            generate_and_save_samples(
                self.generator, self.fixed_noise, epoch=0,
                sample_dir=cfg.SAMPLE_DIR,
            )
            print("[SAMPLE] Saved initial (untrained) samples -> epoch_0000.png")

        # ── Epoch loop ────────────────────────────────────────
        for epoch in range(self.start_epoch, total_epochs):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            num_batches = 0
            epoch_start = time.time()

            for real_imgs, _ in self.dataloader:
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(self.device)

                # Labels
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)

                # ──────────────────────────────────────────────
                # 1.  Train Discriminator
                # ──────────────────────────────────────────────
                self.optimizer_D.zero_grad()

                # Loss on real images
                real_preds = self.discriminator(real_imgs)
                d_loss_real = self.criterion(real_preds, real_labels)

                # Loss on fake images
                z = torch.randn(batch_size, cfg.LATENT_DIM, device=self.device)
                fake_imgs = self.generator(z)
                fake_preds = self.discriminator(fake_imgs.detach())
                d_loss_fake = self.criterion(fake_preds, fake_labels)

                # Total discriminator loss
                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss.backward()
                self.optimizer_D.step()

                # ──────────────────────────────────────────────
                # 2.  Train Generator
                # ──────────────────────────────────────────────
                self.optimizer_G.zero_grad()

                # Generator wants D to think fakes are real
                fake_preds_for_g = self.discriminator(fake_imgs)
                g_loss = self.criterion(fake_preds_for_g, real_labels)

                g_loss.backward()
                self.optimizer_G.step()

                # ── Accumulate batch losses ───────────────────
                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                num_batches += 1

            # ── Epoch-level metrics ───────────────────────────
            avg_d_loss = epoch_d_loss / num_batches
            avg_g_loss = epoch_g_loss / num_batches
            epoch_time = time.time() - epoch_start

            self.metrics_history.append({
                "epoch": epoch + 1,
                "d_loss": round(avg_d_loss, 6),
                "g_loss": round(avg_g_loss, 6),
                "epoch_time_s": round(epoch_time, 2),
            })

            print(
                f"Epoch [{epoch+1:3d}/{total_epochs}]  "
                f"D_loss: {avg_d_loss:.4f}  G_loss: {avg_g_loss:.4f}  "
                f"Time: {epoch_time:.1f}s"
            )

            # ── Periodic sample generation ────────────────────
            if (epoch + 1) % cfg.SAMPLE_INTERVAL == 0 or (epoch + 1) == total_epochs:
                fp = generate_and_save_samples(
                    self.generator, self.fixed_noise, epoch=epoch + 1,
                    sample_dir=cfg.SAMPLE_DIR,
                )
                print(f"[SAMPLE] Saved -> {fp}")

            # ── Checkpoint every SAMPLE_INTERVAL epochs ───────
            if (epoch + 1) % cfg.SAMPLE_INTERVAL == 0 or (epoch + 1) == total_epochs:
                self.save_checkpoint(epoch)

            # ── Save metrics after every epoch ────────────────
            self.save_metrics()

        print(f"\n{'='*60}")
        print("  Training complete.")
        print(f"{'='*60}\n")
