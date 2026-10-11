"""
Quick training test — run 3 epochs to generate real artifacts
for verifying the dashboard displays actual training data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# Override for a quick test
config.NUM_EPOCHS = 3
config.SAMPLE_INTERVAL = 1  # Save samples every epoch

from src.trainer import GANTrainer

trainer = GANTrainer(config)
trainer.train(resume=False)
print("\n[DONE] Quick training completed — check outputs/ for artifacts.")
