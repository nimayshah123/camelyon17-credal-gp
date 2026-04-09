"""
06_domain_selection.py — Standalone: hospital ranking and greedy selection.

Fits a credal GP on source hospitals and evaluates:
  1. Expected credal width reduction from adding each hospital.
  2. Ranks hospitals by this reduction.
  3. Runs greedy multi-round selection to show the improvement curve.
  4. Generates Figure 3 (domain selection + metrics).

Usage
-----
    python scripts/06_domain_selection.py
"""

import os
import sys
import time
import numpy as np
import pickle

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import config
from src.feature_extractor import (
    load_features_from_disk,
    load_labels_from_disk,
    features_exist_on_disk,
)
from src.expansion import get_top_expansion_features
from src.credal_gp import fit_credal_gp_on_top_feature
from src.domain_selection import (
    rank_hospitals_by_reduction,
    greedy_hospital_selection,
    compute_before_after_widths,
)
from src.baselines  import train_all_baselines
from src.metrics    import aggregate_baseline_metrics
from src.visualize  import plot_figure3
import matplotlib.pyplot as plt


def load_or_compute(features_dict, labels_dict):
    """Load pre-computed results or compute fresh."""
    results = {}

    # Metrics
    m_path = os.path.join(config.OUTPUTS_DIR, 'metrics_results.pkl')
    if os.path.exists(m_path):
        with open(m_path, 'rb') as f:
            results['metrics'] = pickle.load(f)
    else:
        results['metrics'] = None

    # Baselines
    bl_path = os.path.join(config.OUTPUTS_DIR, 'baselines.pkl')
    if os.path.exists(bl_path):
        with open(bl_path, 'rb') as f:
            results['baselines'] = pickle.load(f)
    else:
        results['baselines'] = None

    return results


def load_top_feature_idx(features_dict):
    path = os.path.join(config.OUTPUTS_DIR, 'top_feature_idx.npy')
    if os.path.exists(path):
        return int(np.load(path)[0])
    top = get_top_expansion_features(features_dict, n=1)
    return top[0][0]


def main():
    print(f"\n{'='*60}")
    print("STEP 6: DOMAIN SELECTION")
    print(f"{'='*60}")

    if not features_exist_on_disk():
        print("  ERROR: Feature files not found.")
        print("  Run scripts/01_extract_features.py first.")
        sys.exit(1)

    print("  Loading features …")
    features_dict = load_features_from_disk()
    labels_dict   = load_labels_from_disk()

    cached = load_or_compute(features_dict, labels_dict)

    top_feature_idx = load_top_feature_idx(features_dict)
    print(f"  Using PCA-{top_feature_idx} as top expansion feature.")

    # -----------------------------------------------------------------------
    # Fit Credal GP on source hospitals
    # -----------------------------------------------------------------------
    print("\n  Fitting Credal GP on source hospitals …")
    t0 = time.time()
    gp = fit_credal_gp_on_top_feature(
        features_dict, labels_dict, top_feature_idx,
        hospital_ids=config.SOURCE_HOSPITALS,
    )
    print(f"  Fitted in {time.time()-t0:.2f}s")

    # -----------------------------------------------------------------------
    # Rank all hospitals by expected width reduction
    # -----------------------------------------------------------------------
    print("\n  Ranking all 5 hospitals by expected credal width reduction …")
    t1 = time.time()
    reductions = rank_hospitals_by_reduction(
        gp, features_dict, labels_dict,
        top_feature_idx=top_feature_idx,
        query_hospital=config.HELD_OUT_HOSPITAL,
        candidate_hospitals=list(range(config.N_HOSPITALS)),
    )
    print(f"  Ranked in {time.time()-t1:.2f}s")

    print(f"\n  Hospital ranking (by expected credal width reduction):")
    print(f"  {'Rank':<5} {'Hospital':<25} {'Δ width':>10}")
    print("  " + "-" * 43)
    for rank, (h, delta) in enumerate(reductions.items(), 1):
        marker = " ← BEST" if rank == 1 else ""
        print(f"  {rank:<5} {config.HOSPITAL_NAMES[h]:<25} "
              f"{delta:>10.4f}{marker}")

    best_hospital = list(reductions.keys())[0]

    # -----------------------------------------------------------------------
    # Compute before/after widths for the best hospital
    # -----------------------------------------------------------------------
    print(f"\n  Computing before/after width for best hospital H{best_hospital} …")

    all_x = np.concatenate([
        features_dict[h][:, top_feature_idx]
        for h in range(config.N_HOSPITALS)
    ])
    x_min, x_max = all_x.min(), all_x.max()
    margin  = 0.15 * (x_max - x_min)
    X_plot  = np.linspace(x_min - margin, x_max + margin, 200)

    width_before, width_after = compute_before_after_widths(
        gp, features_dict, labels_dict,
        top_feature_idx, best_hospital, X_plot
    )

    mean_before = float(width_before.mean())
    mean_after  = float(width_after.mean())
    pct_red     = (mean_before - mean_after) / (mean_before + 1e-8) * 100
    print(f"  Mean credal width before: {mean_before:.4f}")
    print(f"  Mean credal width after:  {mean_after:.4f}")
    print(f"  Reduction: {pct_red:.1f}%")

    # -----------------------------------------------------------------------
    # Greedy multi-round selection
    # -----------------------------------------------------------------------
    print(f"\n  Running greedy multi-round hospital selection …")
    t2 = time.time()
    selected_order, mean_widths, round_reductions = greedy_hospital_selection(
        features_dict, labels_dict,
        top_feature_idx=top_feature_idx,
        n_rounds=config.N_HOSPITALS - 1,
        query_hospital=config.HELD_OUT_HOSPITAL,
        verbose=True,
    )
    print(f"  Greedy selection done in {time.time()-t2:.2f}s")

    # -----------------------------------------------------------------------
    # Load or compute metrics
    # -----------------------------------------------------------------------
    metrics_results = cached['metrics']
    if metrics_results is None:
        print("\n  Metrics not found — computing …")
        baselines = cached['baselines']
        if baselines is None:
            baselines = train_all_baselines(
                features_dict, labels_dict,
                hospital_ids=config.SOURCE_HOSPITALS
            )
        metrics_results = aggregate_baseline_metrics(
            baselines, features_dict, labels_dict, gp, top_feature_idx
        )

    # -----------------------------------------------------------------------
    # Generate Figure 3
    # -----------------------------------------------------------------------
    print("\n  Generating Figure 3 …")
    fig3 = plot_figure3(
        hospital_reductions=reductions,
        width_before=width_before,
        width_after=width_after,
        X_plot=X_plot,
        metrics_results=metrics_results,
        selected_order=selected_order,
        mean_widths=mean_widths,
    )
    plt.close(fig3)

    # -----------------------------------------------------------------------
    # Save domain selection results
    # -----------------------------------------------------------------------
    ds_results = {
        'reductions': reductions,
        'selected_order': selected_order,
        'mean_widths': mean_widths,
        'round_reductions': round_reductions,
        'best_hospital': best_hospital,
        'width_before': width_before,
        'width_after': width_after,
        'X_plot': X_plot,
    }
    ds_path = os.path.join(config.OUTPUTS_DIR, 'domain_selection_results.pkl')
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    with open(ds_path, 'wb') as f:
        pickle.dump(ds_results, f)
    print(f"  Domain selection results saved → {ds_path}")

    print(f"\n  CONCLUSION:")
    print(f"  Hospital H{best_hospital} ({config.HOSPITAL_NAMES[best_hospital]}) "
          f"provides the greatest credal width reduction.")
    print(f"  Adding it reduces mean credal width at the OOD site by {pct_red:.1f}%,")
    print(f"  confirming the closed-loop active domain selection principle.")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
