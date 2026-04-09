"""
run_all.py — Master pipeline script for the Camelyon17 Credal GP experiment.

Runs the full pipeline in order:
  1. Feature extraction (ResNet50 → PCA 64-d)
  2. Expansion function analysis
  3. Credal GP fitting + Figure 2
  4. Baseline training (ERM, MC Dropout, Deep Ensemble)
  5. Full metrics comparison table
  6. Domain selection + Figure 3
  7. Generate Figure 1 (domain profiles)
  8. Print final summary

Works in both FAST_DEV mode (500 samples/hospital, ~minutes on CPU) and
full dataset mode.

Usage
-----
    python run_all.py [--fast-dev] [--force-extract] [--skip-baselines]

Arguments
---------
--fast-dev       : use 500 samples/hospital for rapid testing.
--force-extract  : re-extract features even if .npy files exist.
--skip-baselines : skip baseline training (useful for re-running GP-only steps).
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on the path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full Camelyon17 Credal GP pipeline."
    )
    parser.add_argument('--fast-dev', action='store_true',
                        help="Use 500 samples/hospital for fast testing.")
    parser.add_argument('--force-extract', action='store_true',
                        help="Re-extract features even if files exist.")
    parser.add_argument('--skip-baselines', action='store_true',
                        help="Skip baseline training/evaluation.")
    return parser.parse_args()


def print_header(step_name: str):
    print(f"\n{'='*60}")
    print(f"{step_name}")
    print(f"{'='*60}")


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"


def main():
    t_start = time.time()
    args    = parse_args()

    if args.fast_dev:
        config.FAST_DEV = True

    print(f"\n{'#'*60}")
    print(f"  CAMELYON17-WILDS: CREDAL GP + EXPANSION FUNCTION")
    print(f"  {'FAST DEV MODE' if config.FAST_DEV else 'FULL DATASET MODE'}")
    print(f"  {'#'*60}")
    print(f"\n  Config:")
    print(f"    Dataset root   : {config.DATASET_ROOT}")
    print(f"    Features dir   : {config.FEATURES_DIR}")
    print(f"    Outputs dir    : {config.OUTPUTS_DIR}")
    print(f"    Held-out hosp  : {config.HELD_OUT_HOSPITAL} "
          f"({config.HOSPITAL_NAMES[config.HELD_OUT_HOSPITAL]})")
    print(f"    PCA dims       : {config.PCA_DIMS}")
    print(f"    GP lengthscales: {config.CREDAL_LENGTHSCALES}")
    print(f"    GP output scales: {config.CREDAL_OUTPUT_SCALES}")
    print(f"    Random seed    : {config.RANDOM_SEED}")

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    os.makedirs(config.DATASET_ROOT, exist_ok=True)

    # =========================================================================
    # STEP 1: Feature Extraction
    # =========================================================================
    print_header("STEP 1: FEATURE EXTRACTION")
    t1 = time.time()

    from src.feature_extractor import (
        load_features_from_disk,
        load_labels_from_disk,
        features_exist_on_disk,
        extract_and_save_all_features,
    )

    features_dict = None
    labels_dict   = None

    if features_exist_on_disk() and not args.force_extract:
        print("  Feature files found — loading from disk.")
        features_dict = load_features_from_disk()
        labels_dict   = load_labels_from_disk()
    else:
        from src.data_loader import load_camelyon17, summarize_splits
        print("  Downloading / loading Camelyon17-WILDS …")
        hospital_dict, _ = load_camelyon17(
            root=config.DATASET_ROOT,
            fast_dev=config.FAST_DEV,
        )
        summarize_splits(hospital_dict)
        features_dict, labels_dict = extract_and_save_all_features(
            hospital_dict, force=args.force_extract
        )

    for h in range(config.N_HOSPITALS):
        print(f"  Hospital {h}: features {features_dict[h].shape}, "
              f"labels {labels_dict[h].shape}")

    print(f"  Step 1 complete — {format_time(time.time()-t1)}")

    # =========================================================================
    # STEP 2: Expansion Function
    # =========================================================================
    print_header("STEP 2: EXPANSION FUNCTION")
    t2 = time.time()

    from src.expansion import (
        compute_all_expansions,
        get_top_expansion_features,
        compute_ood_distance,
    )

    print("  Computing expansion e_φ(z) for all PCA features …")
    expansions = compute_all_expansions(features_dict)
    top_feats  = get_top_expansion_features(features_dict, n=15)

    top_feature_idx = top_feats[0][0]
    top_feature_e   = top_feats[0][1]

    print(f"  Top expansion features:")
    for rank, (f_idx, e_val) in enumerate(top_feats[:5], 1):
        print(f"    {rank}. PCA-{f_idx:3d}  e = {e_val:.4f}")
    print(f"\n  TOP FEATURE: PCA-{top_feature_idx}  (e = {top_feature_e:.4f})")

    # Save top feature index for downstream scripts
    np.save(os.path.join(config.OUTPUTS_DIR, 'top_feature_idx.npy'),
            np.array([top_feature_idx]))
    np.save(os.path.join(config.OUTPUTS_DIR, 'expansion_values.npy'),
            np.array(list(expansions.items())))

    # OOD distances (used in Figure 2 scatter)
    ood_distances = compute_ood_distance(features_dict,
                                          feature_idx=top_feature_idx)

    print(f"  Step 2 complete — {format_time(time.time()-t2)}")

    # =========================================================================
    # STEP 3: Credal GP
    # =========================================================================
    print_header("STEP 3: CREDAL GP FITTING")
    t3 = time.time()

    from src.credal_gp import fit_credal_gp_on_top_feature

    print(f"  Fitting Credal GP on PCA-{top_feature_idx} …")
    print(f"  Kernel set: {len(config.CREDAL_LENGTHSCALES)} × "
          f"{len(config.CREDAL_OUTPUT_SCALES)} = "
          f"{len(config.CREDAL_LENGTHSCALES)*len(config.CREDAL_OUTPUT_SCALES)} "
          f"kernels")

    gp = fit_credal_gp_on_top_feature(
        features_dict, labels_dict, top_feature_idx,
        hospital_ids=config.SOURCE_HOSPITALS,
    )
    print(f"  {gp}")

    # Evaluate credal width at OOD hospital
    ood_h     = config.HELD_OUT_HOSPITAL
    X_ood_1d  = features_dict[ood_h][:, top_feature_idx]
    _, _, _, w_ood = gp.predict(X_ood_1d)

    src_x = np.concatenate([
        features_dict[h][:, top_feature_idx]
        for h in config.SOURCE_HOSPITALS
        if h in features_dict
    ])
    rng = np.random.RandomState(config.RANDOM_SEED)
    idx_src = rng.choice(len(src_x), size=min(500, len(src_x)), replace=False)
    _, _, _, w_src = gp.predict(src_x[idx_src])

    print(f"  Credal width — IID (source): {w_src.mean():.4f}")
    print(f"  Credal width — OOD (H{ood_h}): {w_ood.mean():.4f}")
    print(f"  OOD/IID ratio: {w_ood.mean()/(w_src.mean()+1e-8):.2f}×")

    print(f"  Step 3 complete — {format_time(time.time()-t3)}")

    # =========================================================================
    # STEP 4: Baselines
    # =========================================================================
    baselines = None
    if not args.skip_baselines:
        print_header("STEP 4: BASELINE TRAINING")
        t4 = time.time()

        # Check for cached baselines
        bl_path = os.path.join(config.OUTPUTS_DIR, 'baselines.pkl')
        if os.path.exists(bl_path):
            print("  Loading cached baselines …")
            with open(bl_path, 'rb') as f:
                baselines = pickle.load(f)
        else:
            from src.baselines import train_all_baselines
            baselines = train_all_baselines(
                features_dict, labels_dict,
                hospital_ids=config.SOURCE_HOSPITALS,
                verbose=True,
            )
            with open(bl_path, 'wb') as f:
                pickle.dump(baselines, f)
            print(f"  Baselines saved → {bl_path}")

        print(f"  Step 4 complete — {format_time(time.time()-t4)}")
    else:
        print("\n  [Skipping Step 4: baselines]")
        bl_path = os.path.join(config.OUTPUTS_DIR, 'baselines.pkl')
        if os.path.exists(bl_path):
            with open(bl_path, 'rb') as f:
                baselines = pickle.load(f)

    # =========================================================================
    # STEP 5: Metrics Comparison
    # =========================================================================
    print_header("STEP 5: METRICS COMPARISON")
    t5 = time.time()

    from src.metrics import aggregate_baseline_metrics, print_metrics_table

    metrics_results = None

    m_path = os.path.join(config.OUTPUTS_DIR, 'metrics_results.pkl')
    if os.path.exists(m_path) and not args.skip_baselines:
        # Try loading cached
        with open(m_path, 'rb') as f:
            metrics_results = pickle.load(f)
        print("  Loaded cached metrics results.")
    elif baselines is not None:
        print("  Computing metrics for all methods …")
        metrics_results = aggregate_baseline_metrics(
            baselines, features_dict, labels_dict, gp, top_feature_idx
        )
        with open(m_path, 'wb') as f:
            pickle.dump(metrics_results, f)
    else:
        # Credal GP only
        print("  Computing Credal GP metrics only (baselines skipped) …")
        from src.metrics import (
            compute_all_metrics, make_ood_labels
        )
        X_ood_full = features_dict[ood_h]
        y_ood      = labels_dict[ood_h]

        src_parts_X, src_parts_y = [], []
        for h in config.SOURCE_HOSPITALS:
            src_parts_X.append(features_dict[h])
            src_parts_y.append(labels_dict[h])
        X_iid = np.concatenate(src_parts_X)
        y_iid = np.concatenate(src_parts_y)

        n_ood = len(X_ood_full)
        n_iid = min(n_ood, len(X_iid))
        idx_i = rng.choice(len(X_iid), size=n_iid, replace=False)
        X_iid = X_iid[idx_i];  y_iid = y_iid[idx_i]

        is_ood  = make_ood_labels(n_iid, n_ood)
        X_all   = np.concatenate([X_iid, X_ood_full])
        y_all   = np.concatenate([y_iid, y_ood])
        X_gp_all = X_all[:, top_feature_idx]

        def _sig(x):
            return np.where(x >= 0, 1/(1+np.exp(-x)),
                            np.exp(x)/(1+np.exp(x)))

        c_mean, _, _, c_width = gp.predict(X_gp_all)
        gp_proba = np.clip(_sig(c_mean), 1e-6, 1-1e-6)
        metrics_results = {
            'Credal GP': compute_all_metrics(y_all, gp_proba, c_width, is_ood)
        }

    if metrics_results:
        print_metrics_table(metrics_results)

    print(f"  Step 5 complete — {format_time(time.time()-t5)}")

    # =========================================================================
    # STEP 6: Domain Selection
    # =========================================================================
    print_header("STEP 6: DOMAIN SELECTION")
    t6 = time.time()

    from src.domain_selection import (
        rank_hospitals_by_reduction,
        greedy_hospital_selection,
        compute_before_after_widths,
    )

    print("  Ranking hospitals by expected credal width reduction …")
    reductions = rank_hospitals_by_reduction(
        gp, features_dict, labels_dict,
        top_feature_idx=top_feature_idx,
        query_hospital=config.HELD_OUT_HOSPITAL,
        candidate_hospitals=list(range(config.N_HOSPITALS)),
    )

    best_hospital = list(reductions.keys())[0]
    print(f"\n  Hospital ranking:")
    for rank, (h, delta) in enumerate(reductions.items(), 1):
        marker = " ← BEST" if rank == 1 else ""
        print(f"    {rank}. H{h} ({config.HOSPITAL_NAMES[h]}): "
              f"Δ = {delta:.4f}{marker}")

    # Before/after widths
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

    # Greedy selection
    print("\n  Running greedy multi-round selection …")
    selected_order, mean_widths, round_reductions = greedy_hospital_selection(
        features_dict, labels_dict,
        top_feature_idx=top_feature_idx,
        n_rounds=config.N_HOSPITALS - 1,
        query_hospital=config.HELD_OUT_HOSPITAL,
        verbose=True,
    )

    print(f"  Step 6 complete — {format_time(time.time()-t6)}")

    # =========================================================================
    # STEP 7: Figures
    # =========================================================================
    print_header("STEP 7: GENERATING FIGURES")
    t7 = time.time()

    from src.visualize import plot_all_figures

    top_feat_list = [f for f, _ in get_top_expansion_features(
        features_dict, n=15)]

    plot_all_figures(
        features_dict=features_dict,
        labels_dict=labels_dict,
        expansions=expansions,
        top_feature_indices=top_feat_list,
        credal_gp=gp,
        top_feature_idx=top_feature_idx,
        ood_distances=ood_distances,
        hospital_reductions=reductions,
        width_before=width_before,
        width_after=width_after,
        X_plot=X_plot,
        metrics_results=metrics_results or {'Credal GP': {}},
        selected_order=selected_order,
        mean_widths=mean_widths,
    )

    print(f"  Step 7 complete — {format_time(time.time()-t7)}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time.time() - t_start

    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE — {format_time(total_time)}")
    print(f"  {'#'*60}")

    print(f"\n  Key Results:")
    print(f"    Top expansion feature : PCA-{top_feature_idx} "
          f"(e = {top_feature_e:.4f})")
    print(f"    Credal GP width — IID  : {w_src.mean():.4f}")
    print(f"    Credal GP width — OOD  : {w_ood.mean():.4f}")
    print(f"    OOD/IID width ratio    : "
          f"{w_ood.mean()/(w_src.mean()+1e-8):.2f}×")
    print(f"    Best hospital to add   : H{best_hospital} "
          f"({config.HOSPITAL_NAMES[best_hospital]})")

    if metrics_results:
        print(f"\n  Metrics table:")
        from src.metrics import print_metrics_table
        print_metrics_table(metrics_results)

    print(f"  Figures saved to  : {config.OUTPUTS_DIR}")
    print(f"  Features saved to : {config.FEATURES_DIR}")
    print()


if __name__ == '__main__':
    main()
