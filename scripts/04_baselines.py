"""
04_baselines.py — Standalone: train and evaluate ERM, MC Dropout, Deep Ensemble.

Loads PCA features from disk and:
  1. Trains ERM (LogisticRegression), MC Dropout (N=20 MLPs), and
     Deep Ensemble (N=5 MLPs) on source hospitals.
  2. Evaluates each on the held-out OOD hospital.
  3. Prints predictions summary.

Usage
-----
    python scripts/04_baselines.py
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
from src.baselines import train_all_baselines
from src.metrics import compute_brier, compute_nll, compute_ece


def main():
    print(f"\n{'='*60}")
    print("STEP 4: BASELINE TRAINING & EVALUATION")
    print(f"{'='*60}")

    if not features_exist_on_disk():
        print("  ERROR: Feature files not found.")
        print("  Run scripts/01_extract_features.py first.")
        sys.exit(1)

    print("  Loading features …")
    features_dict = load_features_from_disk()
    labels_dict   = load_labels_from_disk()

    ood_h = config.HELD_OUT_HOSPITAL
    src_hs = config.SOURCE_HOSPITALS

    # -----------------------------------------------------------------------
    # Combine source hospitals for training
    # -----------------------------------------------------------------------
    X_parts, y_parts = [], []
    for h in src_hs:
        X_parts.append(features_dict[h])
        y_parts.append(labels_dict[h])
    X_train = np.concatenate(X_parts)
    y_train = np.concatenate(y_parts)

    print(f"\n  Training set: {len(X_train):,} samples from hospitals {src_hs}")
    print(f"  Test set (OOD, H{ood_h}): "
          f"{len(features_dict[ood_h]):,} samples")

    # -----------------------------------------------------------------------
    # Train baselines
    # -----------------------------------------------------------------------
    t0 = time.time()
    baselines = train_all_baselines(features_dict, labels_dict,
                                     hospital_ids=src_hs, verbose=True)
    print(f"  All baselines trained in {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # Evaluate on OOD test set
    # -----------------------------------------------------------------------
    X_test = features_dict[ood_h]
    y_test = labels_dict[ood_h]

    print(f"\n  Evaluation on OOD hospital {ood_h} "
          f"({config.HOSPITAL_NAMES[ood_h]}):")
    print(f"  {'Method':<25} {'ECE':>8} {'NLL':>8} {'Brier':>8} "
          f"{'Mean_unc':>10}")
    print("  " + "-" * 62)

    for key, bl in baselines.items():
        if key == 'erm':
            proba, unc = bl.predict_proba(X_test)
            unc_val    = float(
                -(proba * np.log(proba + 1e-8) +
                  (1-proba) * np.log(1-proba + 1e-8)).mean()
            )
            unc_mean   = unc_val   # single scalar for ERM
        else:
            proba, unc  = bl.predict_proba(X_test)
            unc_mean    = float(np.mean(unc))

        ece_val   = compute_ece(y_test, proba)
        nll_val   = compute_nll(y_test, proba)
        brier_val = compute_brier(y_test, proba)

        name = bl.name()[:24]
        print(f"  {name:<25} {ece_val:>8.4f} {nll_val:>8.4f} "
              f"{brier_val:>8.4f} {unc_mean:>10.4f}")

    # -----------------------------------------------------------------------
    # Save baseline objects for later use by metrics script
    # -----------------------------------------------------------------------
    import pickle
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    bl_path = os.path.join(config.OUTPUTS_DIR, 'baselines.pkl')
    with open(bl_path, 'wb') as f:
        pickle.dump(baselines, f)
    print(f"\n  Baselines saved → {bl_path}")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
