# GAN Training Progress Visualizer

An interactive experiment dashboard and training pipeline for Generative Adversarial Networks (GANs).

# Academic Context

This project was developed as part of Innovative Assignment II for the Deep Learning Essentials course, under Batch 13.

Assigned Topic: GAN Training Progress Visualizer

Problem Statement: Create an application that records and displays generated images at different stages of GAN training to observe how the generator improves its outputs.

The assignment focuses on applying Generative Adversarial Network (GAN) fundamentals through a practical implementation. This project combines a PyTorch-based GAN trained on the MNIST handwritten-digit dataset with an interactive dashboard for examining generated images and recorded training metrics.

## Objective

Build an application that records and displays generated images at different stages of GAN (Generative Adversarial Network) training to observe how the generator improves its outputs over time.

## Project Structure

```
gan-training-progress-visualizer/
├── app.py              # Streamlit dashboard — interactive visualization
├── train.py            # Entry point — start or resume training
├── config.py           # All hyperparameters and paths
├── requirements.txt    # Python dependencies
├── README.md
├── .gitignore
├── test_smoke.py       # Model and pipeline smoke tests
├── src/
│   ├── __init__.py
│   ├── models.py       # Generator & Discriminator networks
│   ├── dataset.py      # MNIST data loading & preprocessing
│   ├── trainer.py      # Training loop, checkpointing, metrics
│   └── visualization.py# Image-grid saving utilities
├── outputs/
│   ├── samples/        # Generated image grids (epoch_NNNN.png)
│   ├── metrics/        # training_metrics.csv
│   └── checkpoints/    # Resumable model checkpoints (.pt)
└── screenshots/        # Screenshots for the report
```

## Quick Start (Windows)

```powershell
# 1. Navigate to the project
cd "W:\GAN TRAINING\gan-training-progress-visualizer"

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run smoke tests to verify the setup
python test_smoke.py

# 5. Train the GAN (downloads MNIST automatically; auto-resumes if checkpoint exists)
python train.py

# 6. Launch the dashboard
streamlit run app.py
```

> **Tip:** You can launch the dashboard *before* or *during* training — it gracefully handles missing data and includes a refresh button to pick up new results.
> **Resumption:** If training is interrupted, running `python train.py` will automatically detect `outputs/checkpoints/latest_checkpoint.pt` and resume from the next epoch without losing metric history or changing the fixed evaluation noise.

## Dashboard Features

| Section / Tab | Description |
|---|---|
| **Header & Metric Cards** | Live status indicator, Epoch progress (20/20), latest G loss, latest D loss, and average epoch duration |
| **Generated Samples Tab** | Epoch selector slider to inspect 8×8 (64-sample) digit grids, PNG export, and side-by-side longitudinal early vs. late comparison |
| **Training Analytics Tab** | Interactive dark Plotly loss curves (Discriminator in cyan, Generator in violet), theoretical Nash equilibrium reference line (ln 2 ≈ 0.693), range filter, and computation time bar chart |
| **Recorded Metrics Tab** | Full 20-epoch training log with high-precision tabular view and one-click CSV export |
| **Architecture & Setup Tab** | Generator and Discriminator network layer specifications, parameter counts, and artifact filesystem paths |
| **Sidebar** | Hardware execution diagnostic (CPU/GPU), hyperparameters, and dynamic artifact reload button |

The dashboard reads strictly from verified artifacts in `outputs/` without hardcoding or fabricating training metrics.

## How It Works

| Component | Purpose |
|-----------|---------|
| **Generator** | Takes a 100-dimensional noise vector and produces a 28×28 grayscale image |
| **Discriminator** | Takes a 28×28 image and outputs the probability that it is a real MNIST digit |
| **Training loop** | Alternates between training D (to distinguish real from fake) and G (to fool D) |
| **Fixed noise** | The same 64 noise vectors are decoded every few epochs so you can visually compare progress |
| **Checkpointing** | Models, optimizers and metrics are saved periodically; training can resume after interruption |

## Configuration

Edit `config.py` to change hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LATENT_DIM` | 100 | Size of the noise vector |
| `NUM_EPOCHS` | 50 | Total training epochs |
| `BATCH_SIZE` | 64 | Images per mini-batch |
| `LEARNING_RATE` | 0.0002 | Adam learning rate |
| `BETAS` | (0.5, 0.999) | Adam momentum terms |
| `SAMPLE_INTERVAL` | 5 | Epochs between sample snapshots |
| `SEED` | 42 | Random seed for reproducibility |

## Output Artifacts

| Location | Contents |
|----------|----------|
| `outputs/samples/epoch_NNNN.png` | 8×8 grid of generated images at that epoch |
| `outputs/metrics/training_metrics.csv` | CSV with columns: `epoch, d_loss, g_loss, epoch_time_s` |
| `outputs/checkpoints/latest_checkpoint.pt` | Model weights, optimizer states, fixed noise, metrics history |
| `outputs/analysis/loss_and_time_analysis.png` | Analysis plots of loss curves and epoch computation times |

## Technology Stack

- **Training:** Python, PyTorch, torchvision
- **Dashboard:** Streamlit, Plotly
- **Utilities:** NumPy, Pandas, Pillow, Matplotlib
