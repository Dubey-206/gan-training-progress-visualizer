"""
app.py — Premium Streamlit Dashboard for GAN Training Progress Visualizer.

Displays real training metrics, generated sample grids, and interactive
Plotly loss charts. All data is loaded directly from the outputs/ directory
produced by train.py — no fabricated metrics.

Theme: Deep Navy & Charcoal with subtle Violet / Cyan accents.
"""

import os
import sys
import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Paths — resolved relative to this file, not working directory
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SAMPLE_DIR = os.path.join(PROJECT_ROOT, "outputs", "samples")
METRICS_CSV = os.path.join(PROJECT_ROOT, "outputs", "metrics", "training_metrics.csv")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "latest_checkpoint.pt")

try:
    import importlib
    import config as cfg
    importlib.reload(cfg)
    CONFIG_LOADED = True
except ImportError:
    CONFIG_LOADED = False


# ─────────────────────────────────────────────────────────────
# Data Loading & Artifact Discovery
# ─────────────────────────────────────────────────────────────
def discover_sample_files(sample_dir: str) -> list[tuple[int, str]]:
    """
    Scan sample_dir for files matching epoch_NNNN.png and return
    a sorted list of (epoch_number, filepath).
    """
    if not os.path.isdir(sample_dir):
        return []
    results = []
    pattern = re.compile(r"^epoch_(\d{4})\.png$")
    for fname in os.listdir(sample_dir):
        m = pattern.match(fname)
        if m:
            epoch_num = int(m.group(1))
            results.append((epoch_num, os.path.join(sample_dir, fname)))
    results.sort(key=lambda x: x[0])
    return results


def load_metrics() -> pd.DataFrame | None:
    """
    Load the training metrics CSV. Returns None if the file does
    not exist, is empty, or is malformed.
    """
    if not os.path.isfile(METRICS_CSV):
        return None
    try:
        df = pd.read_csv(METRICS_CSV)
        if df.empty or "epoch" not in df.columns:
            return None
        return df
    except Exception:
        return None


def infer_training_status(df: pd.DataFrame | None, total_epochs: int) -> tuple[str, str, str]:
    """
    Returns (status_label, badge_class, description) based strictly
    on available artifacts on disk.
    """
    if df is None or df.empty:
        return "Not Started", "status-not-started", "No training history recorded yet"
    last_epoch = int(df["epoch"].max())
    if last_epoch >= total_epochs:
        return "Completed", "status-completed", f"All {total_epochs} epochs finished"
    return (
        f"In Progress ({last_epoch}/{total_epochs})",
        "status-in-progress",
        f"Completed epoch {last_epoch} of {total_epochs}",
    )


# ─────────────────────────────────────────────────────────────
# Streamlit Page Setup
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GAN Training Progress Visualizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS Design System: AI Research Lab Theme
# Deep navy & charcoal with violet / cyan subtle accents
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #0B0F19;
        --bg-card: #111827;
        --bg-card-subtle: #161F33;
        --border-subtle: #1F293D;
        --border-accent: #2E3B55;
        --accent-cyan: #06B6D4;
        --accent-violet: #8B5CF6;
        --accent-emerald: #10B981;
        --accent-rose: #F43F5E;
        --text-primary: #F3F4F6;
        --text-secondary: #9CA3AF;
        --text-muted: #6B7280;
    }

    /* Base Font & App Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    .stApp {
        background-color: var(--bg-main);
        color: var(--text-primary);
    }

    /* Top Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(22, 31, 51, 0.85) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }

    .hero-badge-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.4rem;
    }

    .hero-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        background: rgba(139, 92, 246, 0.15);
        color: #A78BFA;
        border: 1px solid rgba(139, 92, 246, 0.25);
    }

    .hero-tag-cyan {
        background: rgba(6, 182, 212, 0.15);
        color: #22D3EE;
        border: 1px solid rgba(6, 182, 212, 0.25);
    }

    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #FFFFFF;
        margin: 0.2rem 0 0.5rem 0;
        background: linear-gradient(120deg, #FFFFFF 60%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        font-size: 0.92rem;
        color: var(--text-secondary);
        line-height: 1.55;
        margin: 0;
        max-width: 95%;
    }

    /* Metric Cards Grid */
    .metric-card-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .metric-tile {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }

    .metric-tile:hover {
        border-color: var(--border-accent);
        transform: translateY(-1px);
    }

    .metric-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.45rem;
    }

    .metric-label {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
    }

    .metric-icon {
        font-size: 0.95rem;
        opacity: 0.8;
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.15;
    }

    .metric-subtext {
        font-size: 0.78rem;
        color: var(--text-secondary);
        margin-top: 0.35rem;
    }

    /* Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
    }

    .status-completed {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-in-progress {
        background: rgba(6, 182, 212, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }

    .status-not-started {
        background: rgba(156, 163, 175, 0.15);
        color: #9CA3AF;
        border: 1px solid rgba(156, 163, 175, 0.3);
    }

    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: currentColor;
    }

    /* Notice / Explanatory Callouts */
    .callout-box {
        background: rgba(22, 31, 51, 0.65);
        border: 1px solid var(--border-subtle);
        border-left: 3px solid var(--accent-violet);
        border-radius: 8px;
        padding: 0.85rem 1.15rem;
        font-size: 0.86rem;
        color: #D1D5DB;
        line-height: 1.5;
        margin-bottom: 1.2rem;
    }

    .callout-cyan {
        border-left-color: var(--accent-cyan);
    }

    /* Tab and Container Tweaks */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: transparent;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 0.25rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        font-weight: 600;
        font-size: 0.88rem;
        color: var(--text-secondary);
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.1rem;
    }

    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom: 2px solid var(--accent-cyan) !important;
        background-color: rgba(6, 182, 212, 0.08) !important;
    }

    /* Image Showcase Frame */
    .image-showcase-frame {
        background: #000000;
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.6);
        text-align: center;
        margin-bottom: 0.75rem;
    }

    .image-caption-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0.6rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--text-secondary);
    }

    /* Comparison Headings */
    .compare-card-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #9CA3AF;
        margin-bottom: 0.45rem;
        text-align: center;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0D121F;
        border-right: 1px solid var(--border-subtle);
    }

    /* Custom Streamlit component overrides */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        overflow: hidden;
    }

    /* Cleaner Button Styling */
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        background: #1E293B;
        color: #F8FAFC;
        border: 1px solid var(--border-accent);
        transition: all 0.15s ease;
    }

    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background: #334155;
        border-color: #64748B;
        color: #FFFFFF;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Load Training State
# ─────────────────────────────────────────────────────────────
df = load_metrics()
sample_files = discover_sample_files(SAMPLE_DIR)
total_epochs = cfg.NUM_EPOCHS if CONFIG_LOADED else 20

status_text, status_class, status_detail = infer_training_status(df, total_epochs)

if df is not None and not df.empty:
    last_epoch = int(df["epoch"].max())
    latest_d = df.iloc[-1]["d_loss"]
    latest_g = df.iloc[-1]["g_loss"]
    avg_duration = df["epoch_time_s"].mean() if "epoch_time_s" in df.columns else None
    total_time = df["epoch_time_s"].sum() if "epoch_time_s" in df.columns else None
else:
    last_epoch = 0
    latest_d = None
    latest_g = None
    avg_duration = None
    total_time = None


# ─────────────────────────────────────────────────────────────
# Sidebar Configuration & Hardware Status
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.2rem;">
        <span style="font-size:1.4rem;">⚡</span>
        <h3 style="margin:0; font-size:1.15rem; font-weight:800; color:#FFFFFF;">GAN Inspector</h3>
    </div>
    <div style="font-size:0.75rem; color:#6B7280; font-family:'JetBrains Mono'; margin-bottom:1rem;">
        Adversarial Progress Monitor
    </div>
    """, unsafe_allow_html=True)

    # Refresh Data Action
    if st.button("🔄 Reload Artifacts", use_container_width=True,
                 help="Check filesystem for newly saved sample grids, metrics, or checkpoints"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("<hr style='border:none; border-top:1px solid #1F293D; margin:1.2rem 0;'>", unsafe_allow_html=True)

    # System & Runtime Diagnostic
    st.markdown("<div style='font-size:0.76rem; font-weight:700; text-transform:uppercase; color:#9CA3AF; margin-bottom:0.6rem;'>Execution Device</div>", unsafe_allow_html=True)
    try:
        import torch
        import subprocess

        gpu_hardware_name = None
        try:
            gpu_hardware_name = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            gpu_hardware_name = None

        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            st.markdown(f"""
            <div style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.25); border-radius:8px; padding:0.55rem 0.8rem; font-size:0.8rem; color:#34D399;">
                <strong>GPU Active:</strong> {dev_name}
            </div>
            """, unsafe_allow_html=True)
        elif gpu_hardware_name:
            st.markdown(f"""
            <div style="background:rgba(22,31,51,0.8); border:1px solid #1F293D; border-radius:8px; padding:0.55rem 0.8rem; font-size:0.8rem; color:#94A3B8; line-height:1.45;">
                💻 <strong>CPU Execution</strong><br>
                <span style="font-size:0.73rem; color:#64748B;">Hardware: {gpu_hardware_name} (PyTorch CPU build active)</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(22,31,51,0.8); border:1px solid #1F293D; border-radius:8px; padding:0.55rem 0.8rem; font-size:0.8rem; color:#94A3B8;">
                💻 <strong>CPU Execution</strong>
            </div>
            """, unsafe_allow_html=True)
    except ImportError:
        st.caption("PyTorch not installed in active environment")

    st.markdown("<hr style='border:none; border-top:1px solid #1F293D; margin:1.2rem 0;'>", unsafe_allow_html=True)

    # Verified Hyperparameters
    st.markdown("<div style='font-size:0.76rem; font-weight:700; text-transform:uppercase; color:#9CA3AF; margin-bottom:0.6rem;'>Configuration Parameters</div>", unsafe_allow_html=True)
    if CONFIG_LOADED:
        st.markdown(f"""
        <div style="font-size:0.8rem; line-height:1.7; color:#CBD5E1;">
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748B;">Target Epochs</span>
                <span style="font-family:'JetBrains Mono'; font-weight:600;">{cfg.NUM_EPOCHS}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748B;">Batch Size</span>
                <span style="font-family:'JetBrains Mono'; font-weight:600;">{cfg.BATCH_SIZE}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748B;">Latent Dimension (z)</span>
                <span style="font-family:'JetBrains Mono'; font-weight:600;">{cfg.LATENT_DIM}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748B;">Learning Rate</span>
                <span style="font-family:'JetBrains Mono'; font-weight:600;">{cfg.LEARNING_RATE}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748B;">Adam Betas</span>
                <span style="font-family:'JetBrains Mono'; font-weight:600;">{cfg.BETAS}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748B;">Sample Interval</span>
                <span style="font-family:'JetBrains Mono'; font-weight:600;">Every {cfg.SAMPLE_INTERVAL} ep</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748B;">Random Seed</span>
                <span style="font-family:'JetBrains Mono'; font-weight:600;">{cfg.SEED}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("Default configuration in use.")

    st.markdown("<hr style='border:none; border-top:1px solid #1F293D; margin:1.2rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#6B7280; line-height:1.5;">
        <strong>Artifact Source:</strong> <code>outputs/</code>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Header Banner
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-badge-row">
        <span class="hero-tag">MNIST GAN</span>
        <span class="hero-tag hero-tag-cyan">Minimax Game</span>
        <div class="status-badge {status_class}" style="margin-left:auto;">
            <div class="status-dot"></div>
            {status_text}
        </div>
    </div>
    <div class="hero-title">GAN Training Progress Visualizer</div>
    <p class="hero-description">
        Monitoring an adversarial zero-sum game between a <strong>Generator</strong> (mapping 100-dim noise to 28&times;28 images)
        and a <strong>Discriminator</strong> (evaluating real vs. synthetic digit authenticity).
        Visual sample grids are captured using a frozen latent noise vector of 64 standard normal samples to track
        the step-by-step structural evolution of synthetic handwritten digits across training epochs.
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Primary Metric Overview Grid (4 Cards)
# ─────────────────────────────────────────────────────────────
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown(f"""
    <div class="metric-tile">
        <div class="metric-header">
            <span class="metric-label">Epoch Progress</span>
            <span class="metric-icon">⏱</span>
        </div>
        <div class="metric-value">{last_epoch} <span style="font-size:0.95rem; font-weight:500; color:#6B7280;">/ {total_epochs}</span></div>
        <div class="metric-subtext">{status_detail}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    g_val = f"{latest_g:.4f}" if latest_g is not None else "—"
    st.markdown(f"""
    <div class="metric-tile">
        <div class="metric-header">
            <span class="metric-label">Latest Generator Loss</span>
            <span class="metric-icon">🎨</span>
        </div>
        <div class="metric-value" style="color:#A78BFA;">{g_val}</div>
        <div class="metric-subtext">BCE Loss (-log D(G(z)))</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    d_val = f"{latest_d:.4f}" if latest_d is not None else "—"
    st.markdown(f"""
    <div class="metric-tile">
        <div class="metric-header">
            <span class="metric-label">Latest Discriminator Loss</span>
            <span class="metric-icon">🛡</span>
        </div>
        <div class="metric-value" style="color:#38BDF8;">{d_val}</div>
        <div class="metric-subtext">BCE Loss (Real vs. Fake)</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    duration_str = f"{avg_duration:.1f}s" if avg_duration is not None else "—"
    total_str = f"{(total_time/60):.1f} min total" if total_time is not None else "No metrics"
    st.markdown(f"""
    <div class="metric-tile">
        <div class="metric-header">
            <span class="metric-label">Avg. Epoch Duration</span>
            <span class="metric-icon">⚡</span>
        </div>
        <div class="metric-value">{duration_str}</div>
        <div class="metric-subtext">{total_str}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Main Navigation Tabs
# ─────────────────────────────────────────────────────────────
tab_gallery, tab_analytics, tab_metrics, tab_architecture = st.tabs([
    "🖼️ Generated Samples",
    "📈 Training Analytics",
    "📋 Recorded Metrics",
    "🏗️ Architecture & Setup",
])


# ═════════════════════════════════════════════════════════════
# TAB 1: GENERATED SAMPLES GALLERY & COMPARISON
# ═════════════════════════════════════════════════════════════
with tab_gallery:
    st.markdown("""
    <div class="callout-box">
        <strong>Fixed Latent Noise Baseline (64 vectors in R^100):</strong>
        At initial setup, 64 constant noise vectors were generated and preserved across all checkpoints.
        Every sample grid below is rendered from this exact fixed tensor, meaning each tile location (row, column)
        represents the continuous geometric trajectory of a specific synthetic digit concept over successive epochs.
    </div>
    """, unsafe_allow_html=True)

    if sample_files:
        epoch_numbers = [e for e, _ in sample_files]
        epoch_paths = {e: p for e, p in sample_files}

        # Slider and download header
        col_ctrl, col_dl = st.columns([4, 1])
        with col_ctrl:
            selected_epoch = st.select_slider(
                "Select Training Epoch Snapshot",
                options=epoch_numbers,
                value=epoch_numbers[-1],
                format_func=lambda e: f"Epoch {e:02d}" if e > 0 else "Epoch 00 (Untrained Baseline)",
                key="sample_epoch_slider",
            )
        with col_dl:
            st.markdown("<div style='height:1.7rem;'></div>", unsafe_allow_html=True)
            with open(epoch_paths[selected_epoch], "rb") as f:
                st.download_button(
                    label=f"⬇ Download Epoch {selected_epoch} Grid",
                    data=f.read(),
                    file_name=f"epoch_{selected_epoch:04d}.png",
                    mime="image/png",
                    use_container_width=True,
                )

        # Primary Image Frame
        st.markdown(f"""
        <div class="image-showcase-frame">
            <div class="image-caption-bar">
                <span>SNAPSHOT: EPOCH {selected_epoch} OF {total_epochs}</span>
                <span>8×8 TILE GRID (64 SAMPLES) · 28×28 PX PER DIGIT</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        img = Image.open(epoch_paths[selected_epoch])
        st.image(img, use_container_width=True)

        st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

        # Early vs. Late Comparison View
        if len(sample_files) >= 2:
            st.markdown("""
            <div style="font-size:1.05rem; font-weight:700; color:#FFFFFF; margin-bottom:0.8rem; display:flex; align-items:center; gap:0.5rem;">
                <span>🔀</span> Longitudinal Progression: Baseline vs. Trained
            </div>
            """, unsafe_allow_html=True)

            earliest_epoch, earliest_path = sample_files[0]
            latest_epoch, latest_path = sample_files[-1]

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown(f"""
                <div class="compare-card-title">
                    Initial Baseline · Epoch {earliest_epoch} (Untrained Static)
                </div>
                """, unsafe_allow_html=True)
                st.image(Image.open(earliest_path), use_container_width=True)

            with col_c2:
                st.markdown(f"""
                <div class="compare-card-title">
                    Converged Output · Epoch {latest_epoch} (Learned Digits)
                </div>
                """, unsafe_allow_html=True)
                st.image(Image.open(latest_path), use_container_width=True)

    else:
        st.info("No generated sample grids discovered in `outputs/samples/`. Run training to populate sample images.")


# ═════════════════════════════════════════════════════════════
# TAB 2: PLOTLY TRAINING ANALYTICS
# ═════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("""
    <div class="callout-box callout-cyan">
        <strong>Interpreting Adversarial Loss Trajectories:</strong>
        In standard supervised learning, a decreasing loss curve indicates convergence. In a GAN zero-sum game,
        losses quantify the relative advantage between two adversaries. An oscillating or upward-trending
        Generator loss typically occurs when the Discriminator becomes more selective at penalizing boundary imperfections.
        Always judge model fidelity by visual sample progression rather than standalone loss values.
    </div>
    """, unsafe_allow_html=True)

    if df is not None and not df.empty:
        min_epoch = int(df["epoch"].min())
        max_epoch = int(df["epoch"].max())

        # Range filter
        if max_epoch > min_epoch:
            epoch_range = st.slider(
                "Filter Analyzed Epoch Range",
                min_value=min_epoch,
                max_value=max_epoch,
                value=(min_epoch, max_epoch),
                key="analytics_epoch_range",
            )
            filtered = df[(df["epoch"] >= epoch_range[0]) & (df["epoch"] <= epoch_range[1])]
        else:
            filtered = df

        # Plotly Adversarial Loss Curve
        fig_loss = go.Figure()

        # Discriminator Loss Trace
        fig_loss.add_trace(
            go.Scatter(
                x=filtered["epoch"],
                y=filtered["d_loss"],
                name="Discriminator Loss (D)",
                mode="lines+markers",
                line=dict(width=2.5, color="#38BDF8"),
                marker=dict(size=6, color="#38BDF8", symbol="circle"),
                hovertemplate="<b>Epoch %{x}</b><br>Discriminator Loss: %{y:.4f}<extra></extra>",
            )
        )

        # Generator Loss Trace
        fig_loss.add_trace(
            go.Scatter(
                x=filtered["epoch"],
                y=filtered["g_loss"],
                name="Generator Loss (G)",
                mode="lines+markers",
                line=dict(width=2.5, color="#A78BFA"),
                marker=dict(size=6, color="#A78BFA", symbol="diamond"),
                hovertemplate="<b>Epoch %{x}</b><br>Generator Loss: %{y:.4f}<extra></extra>",
            )
        )

        # Theoretical Equilibrium Reference Line (ln(2) ≈ 0.693)
        fig_loss.add_hline(
            y=0.69315,
            line_dash="dot",
            line_color="rgba(156, 163, 175, 0.45)",
            annotation_text="Theoretical Nash Equilibrium: ln(2) ≈ 0.693",
            annotation_position="bottom right",
            annotation_font=dict(size=10, color="#9CA3AF", family="JetBrains Mono"),
        )

        fig_loss.update_layout(
            title=dict(
                text="Adversarial Loss History Across Epochs",
                font=dict(size=15, color="#F3F4F6", family="Plus Jakarta Sans"),
            ),
            xaxis=dict(
                title="Epoch",
                title_font=dict(color="#9CA3AF", size=12),
                tickfont=dict(color="#9CA3AF", family="JetBrains Mono"),
                gridcolor="#1E293B",
                showgrid=True,
                zeroline=False,
                dtick=max(1, len(filtered) // 15),
            ),
            yaxis=dict(
                title="Binary Cross-Entropy Loss",
                title_font=dict(color="#9CA3AF", size=12),
                tickfont=dict(color="#9CA3AF", family="JetBrains Mono"),
                gridcolor="#1E293B",
                showgrid=True,
                zeroline=False,
            ),
            plot_bgcolor="#0D121F",
            paper_bgcolor="#0D121F",
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="#161F33",
                bordercolor="#2E3B55",
                font=dict(family="JetBrains Mono", size=12, color="#FFFFFF"),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.03,
                xanchor="right",
                x=1,
                font=dict(color="#CBD5E1", size=11),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=55, r=30, t=65, b=45),
            height=390,
        )

        st.plotly_chart(fig_loss, use_container_width=True)

        # Per-Epoch Computation Duration Bar Chart
        if "epoch_time_s" in filtered.columns:
            fig_time = go.Figure()
            fig_time.add_trace(
                go.Bar(
                    x=filtered["epoch"],
                    y=filtered["epoch_time_s"],
                    name="Epoch Duration",
                    marker_color="#06B6D4",
                    opacity=0.85,
                    hovertemplate="<b>Epoch %{x}</b><br>Duration: %{y:.2f}s<extra></extra>",
                )
            )

            fig_time.update_layout(
                title=dict(
                    text="Computation Time Per Epoch",
                    font=dict(size=14, color="#F3F4F6", family="Plus Jakarta Sans"),
                ),
                xaxis=dict(
                    title="Epoch",
                    title_font=dict(color="#9CA3AF", size=12),
                    tickfont=dict(color="#9CA3AF", family="JetBrains Mono"),
                    gridcolor="#1E293B",
                    showgrid=True,
                    dtick=max(1, len(filtered) // 15),
                ),
                yaxis=dict(
                    title="Time (Seconds)",
                    title_font=dict(color="#9CA3AF", size=12),
                    tickfont=dict(color="#9CA3AF", family="JetBrains Mono"),
                    gridcolor="#1E293B",
                    showgrid=True,
                ),
                plot_bgcolor="#0D121F",
                paper_bgcolor="#0D121F",
                margin=dict(l=55, r=30, t=55, b=45),
                height=260,
            )

            st.plotly_chart(fig_time, use_container_width=True)

    else:
        st.info("No training loss records found in `outputs/metrics/training_metrics.csv`.")


# ═════════════════════════════════════════════════════════════
# TAB 3: RECORDED METRICS TABLE & EXPORTS
# ═════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
        <span style="font-size:0.95rem; font-weight:700; color:#FFFFFF;">
            Logged Metric Records (training_metrics.csv)
        </span>
    </div>
    """, unsafe_allow_html=True)

    if df is not None and not df.empty:
        display_df = df.copy()
        display_df.columns = ["Epoch", "Discriminator Loss", "Generator Loss", "Duration (s)"]

        # Styled Table with JetBrains Mono numbers
        st.dataframe(
            display_df.style.format({
                "Discriminator Loss": "{:.6f}",
                "Generator Loss": "{:.6f}",
                "Duration (s)": "{:.2f}",
            }),
            use_container_width=True,
            height=min(450, 36 * len(display_df) + 40),
        )

        col_dl_csv, col_spacer = st.columns([1, 3])
        with col_dl_csv:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Export Metrics CSV",
                data=csv_bytes,
                file_name="training_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("No metric entries logged.")


# ═════════════════════════════════════════════════════════════
# TAB 4: ARCHITECTURE SPECIFICATIONS & DIRECTORY INVENTORY
# ═════════════════════════════════════════════════════════════
with tab_architecture:
    col_arch1, col_arch2 = st.columns(2)

    with col_arch1:
        st.markdown("""
        <div style="background:#111827; border:1px solid #1F293D; border-radius:10px; padding:1.2rem; margin-bottom:1rem;">
            <div style="font-size:0.95rem; font-weight:700; color:#A78BFA; margin-bottom:0.6rem;">
                Generator Network (1,489,424 Parameters)
            </div>
            <div style="font-size:0.84rem; color:#CBD5E1; line-height:1.6;">
                Maps a 100-dimensional continuous noise vector z in R^100 to a 28&times;28 grayscale image.
            </div>
            <div style="margin-top:0.8rem; font-family:'JetBrains Mono'; font-size:0.78rem; color:#94A3B8;">
                • Linear (100 -> 256) + LeakyReLU(0.2)<br>
                • Linear (256 -> 512) + BatchNorm1d + LeakyReLU(0.2)<br>
                • Linear (512 -> 1024) + BatchNorm1d + LeakyReLU(0.2)<br>
                • Linear (1024 -> 784) + Tanh activation<br>
                • Reshape -> (1, 28, 28) with pixel range [-1.0, 1.0]
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_arch2:
        st.markdown("""
        <div style="background:#111827; border:1px solid #1F293D; border-radius:10px; padding:1.2rem; margin-bottom:1rem;">
            <div style="font-size:0.95rem; font-weight:700; color:#38BDF8; margin-bottom:0.6rem;">
                Discriminator Network (533,505 Parameters)
            </div>
            <div style="font-size:0.84rem; color:#CBD5E1; line-height:1.6;">
                Maps a 28&times;28 image x in R^784 to a scalar probability P(real) in [0, 1].
            </div>
            <div style="margin-top:0.8rem; font-family:'JetBrains Mono'; font-size:0.78rem; color:#94A3B8;">
                • Flatten (1, 28, 28) -> 784<br>
                • Linear (784 -> 512) + LeakyReLU(0.2) + Dropout(0.3)<br>
                • Linear (512 -> 256) + LeakyReLU(0.2) + Dropout(0.3)<br>
                • Linear (256 -> 1) + Sigmoid activation
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Artifact Directory Locations Table
    st.markdown("""
    <div style="background:#111827; border:1px solid #1F293D; border-radius:10px; padding:1.2rem;">
        <div style="font-size:0.95rem; font-weight:700; color:#F3F4F6; margin-bottom:0.5rem;">
            📁 Artifact File Registry
        </div>
        <div style="font-size:0.82rem; color:#94A3B8; font-family:'JetBrains Mono'; line-height:1.8;">
            • Metrics Log: <code>outputs/metrics/training_metrics.csv</code><br>
            • Generated Grids: <code>outputs/samples/epoch_NNNN.png</code><br>
            • Resumable Checkpoint: <code>outputs/checkpoints/latest_checkpoint.pt</code><br>
            • Raw MNIST Dataset: <code>data/MNIST/raw/</code>
        </div>
    </div>
    """, unsafe_allow_html=True)
