"""
03_credal_gp_analysis.py — Standalone: fit Credal GP and generate Figure 2.

Loads features from disk, fits the credal GP on source hospitals along the
top expansion feature axis, then:
  1. Plots single GP vs credal GP uncertainty bands.
  2. Plots credal width vs OOD distance scatter with Pearson r.
  3. Saves Figure 2 to outputs/.

Usage
-----
    python scripts/03_credal_gp_analysis.py
"""

import os
import sys
import time
import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import config
from src.feature_extractor import (
    load_features_from_disk,
    load_labels_from_disk,
    features_exist_on_disk,
)
from src.expansion import (
    get_top_expansion_features,
    compute_ood_distance,
)
from src.credal_gp import CredalGP, fit_credal_gp_on_top_feature
from src.visualize import plot_figure2


def load_or_compute_top_feature(features_dict: dict) -> int:
    """Load pre-computed top feature index or recompute."""
    path = os.path.join(config.OUTPUTS_DIR, 'top_feature_idx.npy')
    if os.path.exists(path):
        top_indices = np.load(path)
        return int(top_indices[0])
    else:
        top = get_top_expansion_features(features_dict, n=1)
        return top[0][0]


def main():
    print(f"\n{'='*60}")
    print("STEP 3: CREDAL GP ANALYSIS")
    print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # Load features
    # -----------------------------------------------------------------------
    if not features_exist_on_disk():
        print("  ERROR: Feature files not found.")
        print("  Run scripts/01_extract_features.py first.")
        sys.exit(1)

    print("  Loading features …")
    features_dict = load_features_from_disk()
    labels_dict   = load_labels_from_disk()

    # -----------------------------------------------------------------------
    # Identify top expansion feature
    # -----------------------------------------------------------------------
    top_feature_idx = load_or_compute_top_feature(features_dict)
    print(f"  Using PCA-{top_feature_idx} as top expansion feature.")

    # -----------------------------------------------------------------------
    # Fit Credal GP on source hospitals
    # -----------------------------------------------------------------------
    print(f"\n  Fitting Credal GP on source hospitals "
          f"{config.SOURCE_HOSPITALS} …")
    print(f"  Kernel set: {len(config.CREDAL_LENGTHSCALES)} lengthscales × "
          f"{len(config.CREDAL_OUTPUT_SCALES)} output scales = "
          f"{len(config.CREDAL_LENGTHSCALES)*len(config.CREDAL_OUTPUT_SCALES)}"
          f" kernels")

    t0 = time.time()
    gp = fit_credal_gp_on_top_feature(
        features_dict, labels_dict, top_feature_idx,
        hospital_ids=config.SOURCE_HOSPITALS,
    )
    elapsed = time.time() - t0
    print(f"  Credal GP fitted in {elapsed:.2f}s")
    print(f"  {gp}")

    # -----------------------------------------------------------------------
    # Evaluate on a grid
    # -----------------------------------------------------------------------
    all_x = np.concatenate([
        features_dict[h][:, top_feature_idx]
        for h in range(config.N_HOSPITALS)
    ])
    x_min, x_max = all_x.min(), all_x.max()
    margin  = 0.15 * (x_max - x_min)
    X_grid  = np.linspace(x_min - margin, x_max + margin, 300)

    print(f"\n  Evaluating credal GP on grid [x_min={x_min:.3f}, x_max={x_max:.3f}] …")
    center_mean, lower_mean, upper_mean, credal_width = gp.predict(X_grid)

    # -----------------------------------------------------------------------
    # Compute OOD distances
    # -----------------------------------------------------------------------
    print("  Computing OOD distances …")
    ood_distances = compute_ood_distance(features_dict, feature_idx=top_feature_idx)

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    ood_h   = config.HELD_OUT_HOSPITAL
    X_ood_1d = features_dict[ood_h][:, top_feature_idx]
    _, _, _, width_ood = gp.predict(X_ood_1d)

    src_x = np.concatenate([
        features_dict[h][:, top_feature_idx]
        for h in config.SOURCE_HOSPITALS
        if h in features_dict
    ])
    _, _, _, width_src = gp.predict(src_x[:500])

    print(f"\n  Credal width statistics:")
    print(f"    Source (IID) region:  mean = {width_src.mean():.4f}, "
          f"std = {width_src.std():.4f}")
    print(f"    OOD hospital:         mean = {width_ood.mean():.4f}, "
          f"std = {width_ood.std():.4f}")
    print(f"    OOD / IID ratio:      {width_ood.mean() / (width_src.mean()+1e-8):.2f}×")

    # Pearson r
    from scipy import stats
    n_pts = min(len(X_ood_1d), len(ood_distances), 400)
    rng   = np.random.RandomState(config.RANDOM_SEED)
    idx   = rng.choice(len(X_ood_1d), size=n_pts, replace=False)
    r_val, p_val = stats.pearsonr(ood_distances[idx], width_ood[idx])
    print(f"    Pearson r (width vs OOD distance): r = {r_val:.3f}, "
          f"p = {p_val:.4f}")

    # Save GP for downstream steps
    gp_meta_path = os.path.join(config.OUTPUTS_DIR, 'gp_top_feature_idx.npy')
    np.save(gp_meta_path, np.array([top_feature_idx]))
    print(f"\n  Saved top_feature_idx → {gp_meta_path}")

    # -----------------------------------------------------------------------
    # Generate Figure 2
    # -----------------------------------------------------------------------
    print("\n  Generating Figure 2 …")
    fig2 = plot_figure2(
        features_dict, labels_dict, gp,
        top_feature_idx, ood_distances,
    )
    import matplotlib.pyplot as plt
    plt.close(fig2)

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
