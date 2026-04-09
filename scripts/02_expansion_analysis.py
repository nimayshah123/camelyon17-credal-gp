"""
02_expansion_analysis.py — Standalone: compute and plot the expansion function.

Loads PCA features from disk and:
  1. Computes e_φ(z) = σ²_between / σ²_within for all 64 PCA dimensions.
  2. Identifies top-15 highest-expansion features.
  3. Saves a summary to outputs/ (text + partial Figure 1).
  4. Prints the top features table.

Usage
-----
    python scripts/02_expansion_analysis.py
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
    compute_all_expansions,
    get_top_expansion_features,
    estimate_expansion_from_source_domains,
    per_hospital_feature_means,
)


def main():
    print(f"\n{'='*60}")
    print("STEP 2: EXPANSION FUNCTION ANALYSIS")
    print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # Load features
    # -----------------------------------------------------------------------
    if not features_exist_on_disk():
        print("  ERROR: Feature files not found.")
        print("  Run scripts/01_extract_features.py first.")
        sys.exit(1)

    print("  Loading features from disk …")
    features_dict = load_features_from_disk()
    labels_dict   = load_labels_from_disk()

    for h in range(config.N_HOSPITALS):
        print(f"    Hospital {h}: {features_dict[h].shape}")

    # -----------------------------------------------------------------------
    # Compute expansion (all hospitals)
    # -----------------------------------------------------------------------
    print("\n  Computing expansion function for all PCA features …")
    t0 = time.time()
    expansions_all = compute_all_expansions(features_dict)
    print(f"  Done in {time.time()-t0:.2f}s")

    # -----------------------------------------------------------------------
    # Compute expansion (source hospitals only)
    # -----------------------------------------------------------------------
    print("  Computing expansion using source hospitals only …")
    expansions_src = estimate_expansion_from_source_domains(features_dict)

    # -----------------------------------------------------------------------
    # Top features
    # -----------------------------------------------------------------------
    top_all = get_top_expansion_features(features_dict, n=15)
    top_src = get_top_expansion_features(
        features_dict, n=15,
        hospital_ids=config.SOURCE_HOSPITALS
    )

    print(f"\n  Top-15 expansion features (all 5 hospitals):")
    print(f"  {'Rank':<5} {'PCA dim':<10} {'Expansion e_φ':>15}")
    print("  " + "-" * 33)
    for rank, (feat_idx, e_val) in enumerate(top_all, 1):
        print(f"  {rank:<5} PCA-{feat_idx:<6} {e_val:>15.4f}")

    print(f"\n  Top-15 expansion features (source hospitals only):")
    print(f"  {'Rank':<5} {'PCA dim':<10} {'Expansion e_φ':>15}")
    print("  " + "-" * 33)
    for rank, (feat_idx, e_val) in enumerate(top_src, 1):
        print(f"  {rank:<5} PCA-{feat_idx:<6} {e_val:>15.4f}")

    # -----------------------------------------------------------------------
    # Per-hospital means for top features
    # -----------------------------------------------------------------------
    top_feat_indices = [f for f, _ in top_all[:8]]
    means_mat = per_hospital_feature_means(features_dict, top_feat_indices)

    print(f"\n  Per-hospital means for top-8 expansion features:")
    header = f"  {'Hospital':<25}"
    for f in top_feat_indices:
        header += f" {'PC'+str(f):>8}"
    print(header)
    print("  " + "-" * (25 + 9 * len(top_feat_indices)))
    for h in range(config.N_HOSPITALS):
        row = f"  {config.HOSPITAL_NAMES[h]:<25}"
        for col in range(len(top_feat_indices)):
            row += f" {means_mat[h, col]:>8.3f}"
        print(row)

    # -----------------------------------------------------------------------
    # Save expansion values to disk
    # -----------------------------------------------------------------------
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    exp_path = os.path.join(config.OUTPUTS_DIR, 'expansion_values.npy')
    np.save(exp_path, np.array(list(expansions_all.items())))
    print(f"\n  Expansion values saved → {exp_path}")

    top_idx_path = os.path.join(config.OUTPUTS_DIR, 'top_feature_idx.npy')
    np.save(top_idx_path, np.array([f for f, _ in top_all]))
    print(f"  Top feature indices saved → {top_idx_path}")

    print(f"\n  TOP EXPANSION FEATURE: PCA-{top_all[0][0]}  (e = {top_all[0][1]:.4f})")
    print(f"  This feature will be used as the 1-D credal GP input in Step 3.")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
