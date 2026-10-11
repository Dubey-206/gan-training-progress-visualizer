"""
train.py — Entry point to start (or resume) GAN training.

Usage
-----
    python train.py

The script reads all hyperparameters from config.py, downloads MNIST
if needed, and runs the training loop.  Progress is printed to the
console, and samples / metrics / checkpoints are saved to outputs/.
"""

import config
from src.trainer import GANTrainer


def main():
    trainer = GANTrainer(config)
    trainer.train(resume=True)


if __name__ == "__main__":
    main()
