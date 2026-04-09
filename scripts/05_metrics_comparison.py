"""
05_metrics_comparison.py — Standalone: full metrics comparison table.

Loads baselines from disk, fits a fresh credal GP, evaluates all methods on
the OOD hospital, and prints the complete metrics table (ECE, NLL, Brier,
AUROC, FPR@95).

Usage
-----
    python scripts/05_metrics_comparison.py
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
from src.expansion  import get_top_expansion_features
from src.credal_gp  import fit_credal_gp_on_top_feature
from src.metrics    import aggregate_baseline_metrics, print_metrics_table
from src.baselines  import train_all_baselines


def load_baselines(features_dict, labels_dict):
    """Load saved baselines or retrain if not found."""
    bl_path = os.path.join(config.OUTPUTS_DIR, 'baselines.pkl')
    if os.path.exists(bl_path):
        with open(bl_path, 'rb') as f:
            baselines = pickle.load(f)
        print(f"  Baselines loaded from {bl_path}")
        return baselines
    else:
        print("  Baselines not found — retraining …")
        return train_all_baselines(features_dict, labels_dict,
                                   hospital_ids=config.SOURCE_HOSPITALS)


def load_top_feature_idx(features_dict):
    path = os.path.join(config.OUTPUTS_DIR, 'top_feature_idx.npy')
    if os.path.exists(path):
        return int(np.load(path)[0])
    top = get_top_expansion_features(features_dict, n=1)
    return top[0][0]


def main():
    print(f"\n{'='*60}")
    print("STEP 5: FULL METRICS COMPARISON")
    print(f"{'='*60}")

    if not features_exist_on_disk():
        print("  ERROR: Feature files not found.")
        print("  Run scripts/01_extract_features.py first.")
        sys.exit(1)

    print("  Loading features …")
    features_dict = load_features_from_disk()
    labels_dict   = load_labels_from_disk()

    # -----------------------------------------------------------------------
    # Load/train baselines
    # -----------------------------------------------------------------------
    baselines = load_baselines(features_dict, labels_dict)

    # -----------------------------------------------------------------------
    # Fit Credal GP
    # -----------------------------------------------------------------------
    top_feature_idx = load_top_feature_idx(features_dict)
    print(f"  Using PCA-{top_feature_idx} as top expansion feature.")

    print("  Fitting Credal GP …")
    t0 = time.time()
    gp = fit_credal_gp_on_top_feature(
        features_dict, labels_dict, top_feature_idx,
        hospital_ids=config.SOURCE_HOSPITALS,
    )
    print(f"  Fitted in {time.time()-t0:.2f}s")

    # -----------------------------------------------------------------------
    # Compute all metrics
    # -----------------------------------------------------------------------
    print("\n  Computing metrics for all methods …")
    t1 = time.time()
    results = aggregate_baseline_metrics(
        baselines, features_dict, labels_dict, gp, top_feature_idx
    )
    print(f"  Metrics computed in {time.time()-t1:.2f}s")

    # -----------------------------------------------------------------------
    # Print table
    # -----------------------------------------------------------------------
    print_metrics_table(results)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    results_path = os.path.join(config.OUTPUTS_DIR, 'metrics_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  Results saved → {results_path}")

    # -----------------------------------------------------------------------
    # Highlight key findings
    # -----------------------------------------------------------------------
    if 'Credal GP' in results and 'ERM' in results:
        gp_auroc  = results['Credal GP'].get('auroc', float('nan'))
        erm_auroc = results['ERM'].get('auroc', float('nan'))
        gp_ece    = results['Credal GP'].get('ece', float('nan'))
        erm_ece   = results['ERM'].get('ece', float('nan'))
        print(f"  Key finding — OOD detection AUROC:")
        print(f"    Credal GP: {gp_auroc:.4f}")
        print(f"    ERM:       {erm_auroc:.4f}")
        print(f"    Δ AUROC:   {gp_auroc - erm_auroc:+.4f}")
        print(f"\n  Key finding — ECE (calibration):")
        print(f"    Credal GP: {gp_ece:.4f}")
        print(f"    ERM:       {erm_ece:.4f}")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
