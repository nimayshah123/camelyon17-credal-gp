"""
config.py — Central configuration for Camelyon17 Credal GP pipeline.

All hyperparameters, paths, and experiment settings live here.
Edit FAST_DEV = True for quick iteration (500 samples/hospital).
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "data")
FEATURES_DIR = os.path.join(BASE_DIR, "features")
OUTPUTS_DIR  = os.path.join(BASE_DIR, "outputs")

# ---------------------------------------------------------------------------
# Experiment mode
# ---------------------------------------------------------------------------
FAST_DEV = False          # True → 500 samples/hospital for rapid testing
FAST_DEV_N = 500          # samples per hospital in fast-dev mode

# ---------------------------------------------------------------------------
# Domain setup
# ---------------------------------------------------------------------------
N_HOSPITALS      = 5      # hospitals 0–4
HELD_OUT_HOSPITAL = 3     # RST hospital held out as OOD test domain
HOSPITAL_NAMES   = {
    0: "Radboud (H0)",
    1: "CWZ (H1)",
    2: "UMCU (H2)",
    3: "RST (H3) [OOD]",
    4: "LPON (H4)",
}
SOURCE_HOSPITALS = [h for h in range(N_HOSPITALS) if h != HELD_OUT_HOSPITAL]

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
PCA_DIMS   = 64    # PCA dimensions after ResNet50 2048-d embeddings
BATCH_SIZE = 64    # batch size for ResNet50 forward passes

# ---------------------------------------------------------------------------
# Credal GP
# ---------------------------------------------------------------------------
CREDAL_LENGTHSCALES  = [0.1, 0.3, 0.7, 1.5, 3.5]   # diverse RBF lengthscales
CREDAL_OUTPUT_SCALES = [0.5, 1.0, 2.0]               # diverse output scales
NOISE = 0.05                                           # observation noise (fixed)
GP_SUBSAMPLE = 300   # max training points for GP (kernel matrix memory)

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
N_ENSEMBLE    = 5    # number of models in deep ensemble
N_MC_SAMPLES  = 20   # MC dropout forward passes

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
ECE_BINS = 10

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
DPI = 150
PALETTE = {
    "credal":   "#2E86AB",   # blue  — credal GP
    "single":   "#E84855",   # red   — single GP
    "erm":      "#3BB273",   # green — ERM baseline
    "mc":       "#F18F01",   # orange— MC Dropout
    "ensemble": "#7B2D8B",   # purple— Deep Ensemble
    "ood":      "#FF6B6B",   # coral — OOD region
    "iid":      "#A8DADC",   # teal  — IID region
    "hospitals": ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"],
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
